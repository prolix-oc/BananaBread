"""Bounded request admission and fair, dynamically batched embedding inference."""

import asyncio
import math
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import HTTPException

from bananabread.utils import ExecutionTiming, run_in_threadpool_with_executor


@dataclass
class _Reservation:
    admission: Any
    username: str
    tokens: int = 0
    closed: bool = False

    def reserve_tokens(self, tokens: int):
        self.admission.reserve_tokens(self, tokens)

    async def wait_for_tokens(self, tokens: int, timeout_s: float):
        await self.admission.wait_for_tokens(self, tokens, timeout_s)


class InferenceAdmission:
    """Count admitted requests before accounting, decoding or executor submission."""

    def __init__(
        self, max_requests: int, per_user: int, *, max_tokens=65536, per_user_tokens=32768
    ):
        self.max_requests = max_requests
        self.per_user = per_user
        self.users: dict[str, int] = {}
        self.active = 0
        self.max_tokens = max_tokens
        self.per_user_tokens = per_user_tokens
        self.tokens = 0
        self.user_tokens: dict[str, int] = {}
        self.token_rate: float | None = None
        self.request_rate: float | None = None
        self.waiters: dict[asyncio.Future, asyncio.AbstractEventLoop] = {}
        self.lock = threading.Lock()

    def _overloaded(self, reason):
        # Before the first sample, estimate known work only. A guessed fixed
        # cost per request overstates small bursts and persists under warm load.
        seconds = self.tokens / (self.token_rate or 8192)
        if self.request_rate is not None:
            seconds = max(seconds, self.active / self.request_rate)
        return HTTPException(
            status_code=429,
            detail=f"Inference capacity exceeded; retry later ({reason})",
            headers={"Retry-After": str(max(1, min(60, math.ceil(seconds))))},
        )

    def reserve_tokens(self, reservation: _Reservation, tokens: int):
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        with self.lock:
            self._reserve_locked(reservation, tokens)

    def _reserve_locked(self, reservation, tokens):
        if reservation.closed or reservation.tokens:
            raise RuntimeError("Invalid inference reservation")
        weight = max(1, tokens)
        user_tokens = self.user_tokens.get(reservation.username, 0)
        # Keep large single requests supported, but never queue more work
        # alongside one that exceeds the entire budget.
        if self.tokens and self.tokens + weight > self.max_tokens:
            raise self._overloaded("global token budget")
        if user_tokens and user_tokens + weight > self.per_user_tokens:
            raise self._overloaded("per-user token budget")
        reservation.tokens = weight
        self.tokens += weight
        self.user_tokens[reservation.username] = user_tokens + weight

    async def wait_for_tokens(self, reservation, tokens, timeout_s):
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            with self.lock:
                try:
                    self._reserve_locked(reservation, tokens)
                    return
                except HTTPException:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise
                    # Registration and capacity checking are atomic with release.
                    future = loop.create_future()
                    self.waiters[future] = loop
            try:
                await asyncio.wait_for(future, remaining)
            except TimeoutError:
                with self.lock:
                    raise self._overloaded("token capacity wait timed out")
            finally:
                with self.lock:
                    self.waiters.pop(future, None)

    @staticmethod
    def _wake(future):
        if not future.done():
            future.set_result(None)

    def observe(self, tokens: int, service_s: float, *, workers=1, requests=1.0):
        if service_s <= 0:
            return
        with self.lock:
            if tokens > 0:
                rate = tokens * workers / service_s
                self.token_rate = (
                    rate if self.token_rate is None else self.token_rate * 0.8 + rate * 0.2
                )
            if requests > 0:
                rate = requests * workers / service_s
                self.request_rate = (
                    rate if self.request_rate is None else self.request_rate * 0.8 + rate * 0.2
                )

    @contextmanager
    def slot(self, username: str):
        with self.lock:
            if self.active >= self.max_requests:
                raise self._overloaded("global request limit")
            if self.users.get(username, 0) >= self.per_user:
                raise self._overloaded("per-user request limit")
            self.active += 1
            self.users[username] = self.users.get(username, 0) + 1
            reservation = _Reservation(self, username)
        try:
            yield reservation
        finally:
            with self.lock:
                self.active -= 1
                self.users[username] -= 1
                if not self.users[username]:
                    del self.users[username]
                self.tokens -= reservation.tokens
                if reservation.tokens:
                    self.user_tokens[username] -= reservation.tokens
                    if not self.user_tokens[username]:
                        del self.user_tokens[username]
                reservation.closed = True
                for future, loop in self.waiters.items():
                    loop.call_soon_threadsafe(self._wake, future)

    def stats(self):
        with self.lock:
            return {
                "active_requests": self.active,
                "max_requests": self.max_requests,
                "max_requests_per_user": self.per_user,
                "users": dict(self.users),
                "active_tokens": self.tokens,
                "max_tokens": self.max_tokens,
                "max_tokens_per_user": self.per_user_tokens,
                "user_tokens": dict(self.user_tokens),
                "estimated_tokens_per_second": self.token_rate,
                "estimated_requests_per_second": self.request_rate,
                "waiting_requests": len(self.waiters),
            }


@dataclass(eq=False)
class _Request:
    username: str
    inputs: list[Any]
    lengths: list[int | None]
    input_type: str
    future: asyncio.Future
    timing: ExecutionTiming
    submitted: float = field(default_factory=time.perf_counter)
    offset: int = 0
    outputs: list = field(default_factory=list)
    running: bool = False
    cancelled: bool = False


class EmbeddingScheduler:
    """One bounded inference batch per turn, round-robin by tenant and request.

    Only compatible query/document inputs with known padded lengths can merge.
    Media/unknown preprocessing runs alone. Model selection and encode remain on
    the dedicated executor, with at most one submission per scheduler worker.
    """

    def __init__(
        self,
        executor,
        encode,
        *,
        workers=1,
        batch_size=8,
        max_tokens=8192,
        wait_ms=25,
        quiet_ms=5,
        observe=None,
    ):
        self.executor = executor
        self.encode = encode
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.wait_s = wait_ms / 1000
        self.quiet_s = quiet_ms / 1000
        self.last_arrival = 0.0
        self.collectors: set[asyncio.Future] = set()
        self.observe = observe
        self.batches = 0
        self.encoded_inputs = 0
        self.collection_waits = 0
        self.inference_s = 0.0
        self.known_input_tokens = 0
        self.padded_tokens = 0
        self.ready: OrderedDict[str, deque] = OrderedDict()
        self.wake = asyncio.Event()
        self.closed = False
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(workers)]

    def _enqueue(self, request):
        self.ready.setdefault(request.username, deque()).append(request)
        self.last_arrival = time.perf_counter()
        self._notify_collectors()
        self.wake.set()

    def _notify_collectors(self):
        for future in self.collectors:
            if not future.done():
                future.set_result(None)

    def _remove(self, request):
        queue = self.ready.get(request.username)
        if queue is not None and request in queue:
            queue.remove(request)
            if not queue:
                del self.ready[request.username]
            self._notify_collectors()

    async def submit(self, username, inputs, lengths, input_type, *, timing=None):
        if self.closed:
            raise RuntimeError("Embedding scheduler is closed")
        request = _Request(
            username,
            inputs,
            lengths,
            input_type,
            asyncio.get_running_loop().create_future(),
            timing if timing is not None else ExecutionTiming(),
        )
        self._enqueue(request)
        try:
            return await asyncio.shield(request.future)
        except asyncio.CancelledError:
            request.cancelled = True
            self._remove(request)
            if not request.running:
                request.future.cancel()
            # Retain the admission slot until an already running encode finishes.
            while not request.future.done():
                try:
                    await asyncio.shield(request.future)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not request.future.cancelled():
                request.future.exception()
            raise

    def _batch(self):
        selected, inputs, longest, input_type = [], [], 0, None
        while self.ready and len(inputs) < self.batch_size:
            # A non-fitting tenant keeps its place for the next batch; scan
            # other heads to use the remaining capacity without starving it.
            choice = None
            for username, queue in self.ready.items():
                request = queue[0]
                if input_type is not None and request.input_type != input_type:
                    continue
                length = request.lengths[request.offset]
                if inputs and (
                    length is None or max(longest, length) * (len(inputs) + 1) > self.max_tokens
                ):
                    continue
                choice = username, request
                break
            if choice is None:
                break
            username, request = choice
            queue = self.ready.pop(username)
            queue.popleft()
            if queue:
                self.ready[username] = queue
            input_type = request.input_type
            start = request.offset
            # A request may fill the remaining batch; other users get the next turn.
            while request.offset < len(request.inputs) and len(inputs) < self.batch_size:
                length = request.lengths[request.offset]
                if inputs and (
                    length is None or max(longest, length) * (len(inputs) + 1) > self.max_tokens
                ):
                    break
                inputs.append(request.inputs[request.offset])
                request.offset += 1
                longest = max(longest, length or 0)
                if length is None:
                    break
            request.running = True
            request.timing.queue_s += time.perf_counter() - request.submitted
            selected.append((request, request.offset - start))
            if request.lengths[request.offset - 1] is None:
                break
        return selected, inputs, input_type

    def _collection_delay(self):
        first = next(iter(self.ready.values()))[0]
        length = first.lengths[first.offset]
        if first.offset or length is None or length >= self.max_tokens:
            return 0
        # Bulk work and a full queue already have enough inputs. Collection is
        # useful only for a new, underfilled text batch.
        count = 0
        for queue in self.ready.values():
            for request in queue:
                if request.input_type == first.input_type:
                    count += len(request.inputs) - request.offset
                    if count >= self.batch_size:
                        return 0
        now = time.perf_counter()
        delay = self.wait_s - (now - first.submitted)
        if self.quiet_s:
            delay = min(delay, self.quiet_s - (now - self.last_arrival))
        return max(0, delay)

    async def _collect(self):
        waited = False
        while self.ready and not self.closed:
            delay = self._collection_delay()
            if not delay:
                return
            if not waited:
                self.collection_waits += 1
                waited = True
            future = asyncio.get_running_loop().create_future()
            self.collectors.add(future)
            try:
                # New arrivals wake collection immediately: a full batch need
                # not wait for a timer, and a trickle may extend only to the cap.
                await asyncio.wait_for(future, delay)
            except TimeoutError:
                pass
            finally:
                self.collectors.discard(future)

    def stats(self):
        requests = [request for queue in self.ready.values() for request in queue]
        return {
            "queued_requests": len(requests),
            "queued_inputs": sum(len(request.inputs) - request.offset for request in requests),
            "batches": self.batches,
            "encoded_inputs": self.encoded_inputs,
            "collection_waits": self.collection_waits,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
            "wait_ms": self.wait_s * 1000,
            "quiet_ms": self.quiet_s * 1000,
            "inference_seconds": self.inference_s,
            "known_input_tokens": self.known_input_tokens,
            "padded_tokens": self.padded_tokens,
        }

    async def _worker(self):
        while True:
            while not self.ready:
                if self.closed:
                    return
                self.wake.clear()
                await self.wake.wait()
            await self._collect()
            if not self.ready:
                continue
            selected, inputs, input_type = self._batch()
            timing = ExecutionTiming()
            try:
                output = await run_in_threadpool_with_executor(
                    self.executor, self.encode, inputs, input_type, timing, timing=timing
                )
                if len(output) != len(inputs):
                    raise RuntimeError(
                        f"Embedding model returned {len(output)} vectors; expected {len(inputs)}"
                    )
                lengths = [
                    length
                    for request, count in selected
                    for length in request.lengths[request.offset - count : request.offset]
                ]
                if self.observe is not None:
                    work = sum(lengths) if all(length is not None for length in lengths) else 0
                    equivalents = sum(count / len(request.inputs) for request, count in selected)
                    self.observe(work, timing.work_s, requests=equivalents)
                self.batches += 1
                self.encoded_inputs += len(inputs)
                self.inference_s += timing.work_s
                if all(length is not None for length in lengths):
                    self.known_input_tokens += sum(lengths)
                    self.padded_tokens += max(lengths) * len(lengths)
            except Exception as exc:
                for request, _ in selected:
                    request.running = False
                    if not request.future.done():
                        request.future.set_exception(exc)
                continue
            offset = 0
            for request, count in selected:
                request.running = False
                request.timing.queue_s += timing.queue_s
                request.timing.run_s += timing.run_s
                request.timing.lock_wait_s += timing.lock_wait_s
                if request.cancelled:
                    request.future.cancel()
                else:
                    request.outputs.append(output[offset : offset + count])
                    if request.offset == len(request.inputs):
                        request.future.set_result(np.concatenate(request.outputs, axis=0))
                    else:
                        request.submitted = time.perf_counter()
                        self._enqueue(request)
                offset += count

    async def close(self):
        self.closed = True
        # Drain admitted work before the executors/models are torn down.
        self.wake.set()
        self._notify_collectors()
        await asyncio.gather(*self.tasks)
