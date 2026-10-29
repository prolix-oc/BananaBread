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
        self.lock = threading.Lock()

    def _overloaded(self):
        # A bounded estimate of the current drain time, including a conservative
        # request-count fallback before inference has produced a rate sample.
        seconds = max(self.active * 0.1, self.tokens / (self.token_rate or 8192))
        return HTTPException(
            status_code=429,
            detail="Inference capacity exceeded; retry later",
            headers={"Retry-After": str(max(1, min(60, math.ceil(seconds))))},
        )

    def reserve_tokens(self, reservation: _Reservation, tokens: int):
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        with self.lock:
            if reservation.closed or reservation.tokens:
                raise RuntimeError("Invalid inference reservation")
            weight = max(1, tokens)
            user_tokens = self.user_tokens.get(reservation.username, 0)
            # Keep large single requests supported, but never queue more work
            # alongside one that exceeds the entire budget.
            if (self.tokens and self.tokens + weight > self.max_tokens) or (
                user_tokens and user_tokens + weight > self.per_user_tokens
            ):
                raise self._overloaded()
            reservation.tokens = weight
            self.tokens += weight
            self.user_tokens[reservation.username] = user_tokens + weight

    def observe(self, tokens: int, service_s: float, *, workers=1):
        if tokens <= 0 or service_s <= 0:
            return
        rate = tokens * workers / service_s
        with self.lock:
            self.token_rate = (
                rate if self.token_rate is None else self.token_rate * 0.8 + rate * 0.2
            )

    @contextmanager
    def slot(self, username: str):
        with self.lock:
            if self.active >= self.max_requests or self.users.get(username, 0) >= self.per_user:
                raise self._overloaded()
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
        self, executor, encode, *, workers=1, batch_size=8, max_tokens=8192, wait_ms=2, observe=None
    ):
        self.executor = executor
        self.encode = encode
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.wait_s = wait_ms / 1000
        self.observe = observe
        self.batches = 0
        self.encoded_inputs = 0
        self.collection_waits = 0
        self.ready: OrderedDict[str, deque] = OrderedDict()
        self.wake = asyncio.Event()
        self.closed = False
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(workers)]

    def _enqueue(self, request):
        self.ready.setdefault(request.username, deque()).append(request)
        self.wake.set()

    def _remove(self, request):
        queue = self.ready.get(request.username)
        if queue is not None and request in queue:
            queue.remove(request)
            if not queue:
                del self.ready[request.username]

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
        return max(0, self.wait_s - (time.perf_counter() - first.submitted))

    def stats(self):
        requests = [request for queue in self.ready.values() for request in queue]
        return {
            "queued_requests": len(requests),
            "queued_inputs": sum(len(request.inputs) - request.offset for request in requests),
            "batches": self.batches,
            "encoded_inputs": self.encoded_inputs,
            "collection_waits": self.collection_waits,
        }

    async def _worker(self):
        while True:
            while not self.ready:
                if self.closed:
                    return
                self.wake.clear()
                await self.wake.wait()
            delay = self._collection_delay()
            if delay:
                self.collection_waits += 1
                await asyncio.sleep(delay)
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
                if self.observe is not None and all(length is not None for length in lengths):
                    self.observe(sum(lengths), timing.work_s)
                self.batches += 1
                self.encoded_inputs += len(inputs)
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
        await asyncio.gather(*self.tasks)
