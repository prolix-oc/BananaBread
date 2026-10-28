import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import bananabread.utils as utils
from bananabread.utils import ExecutionTiming, run_in_threadpool_with_executor, timed_lock


@pytest.mark.parametrize("fail", [False, True])
def test_worker_timing_separates_queue_lock_and_work(monkeypatch, fail):
    # A controlled clock keeps the timing assertions independent of machine load.
    clock = iter([0.0, 4.0, 6.0, 9.0, 15.0])
    monkeypatch.setattr(utils, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    timing = ExecutionTiming()
    lock = threading.Lock()
    loop_thread = threading.get_ident()

    def work(value, *, extra):
        assert threading.get_ident() != loop_thread
        with timed_lock(lock, timing):
            if fail:
                raise ValueError("worker failed")
            return value + extra

    async def run(executor):
        return await run_in_threadpool_with_executor(executor, work, 2, extra=3, timing=timing)

    with ThreadPoolExecutor(max_workers=1) as executor:
        if fail:
            with pytest.raises(ValueError, match="worker failed"):
                asyncio.run(run(executor))
        else:
            assert asyncio.run(run(executor)) == 5

    assert timing.queue_s == 4.0
    assert timing.lock_wait_s == 3.0
    assert timing.run_s == 11.0
    assert timing.work_s == 8.0
    assert lock.acquire(blocking=False)
    lock.release()
