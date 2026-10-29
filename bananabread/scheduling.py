"""Bounded request admission and fair, dynamically batched embedding inference."""

import asyncio
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import HTTPException

from bananabread.utils import ExecutionTiming, run_in_threadpool_with_executor


class InferenceAdmission:
    """Count admitted requests before accounting, decoding or executor submission."""

    def __init__(self, max_requests: int, per_user: int):
        self.max_requests = max_requests
        self.per_user = per_user
        self.users: dict[str, int] = {}
        self.active = 0
        self.lock = threading.Lock()

    @contextmanager
    def slot(self, username: str):
        with self.lock:
            if self.active >= self.max_requests or self.users.get(username, 0) >= self.per_user:
                raise HTTPException(
                    status_code=429,
                    detail="Inference capacity exceeded; retry later",
                    headers={"Retry-After": "1"},
                )
            self.active += 1
            self.users[username] = self.users.get(username, 0) + 1
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1
                self.users[username] -= 1
                if not self.users[username]:
                    del self.users[username]

    def stats(self):
        with self.lock:
            return {
                "active_requests": self.active,
                "max_requests": self.max_requests,
                "max_requests_per_user": self.per_user,
                "users": dict(self.users),
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

    def __init__(self, executor, encode, *, workers=1, batch_size=8, max_tokens=8192, wait_ms=2):
        self.executor = executor
        self.encode = encode
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.wait_s = wait_ms / 1000
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
            username = next(iter(self.ready))
            request = self.ready[username][0]
            if input_type is not None and request.input_type != input_type:
                break
            length = request.lengths[request.offset]
            if inputs and (
                length is None or max(longest, length) * (len(inputs) + 1) > self.max_tokens
            ):
                break
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
            if length is None:
                break
        return selected, inputs, input_type

    async def _worker(self):
        while True:
            while not self.ready:
                if self.closed:
                    return
                self.wake.clear()
                await self.wake.wait()
            if self.wait_s:
                await asyncio.sleep(self.wait_s)
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
