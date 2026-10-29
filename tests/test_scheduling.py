"""Concurrency behavior with controlled workers; no model downloads or GPU needed."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi import HTTPException

from bananabread.scheduling import EmbeddingScheduler, InferenceAdmission
from bananabread.utils import run_in_threadpool_with_executor


@pytest.mark.asyncio
async def test_cross_request_batching_preserves_each_response_and_limits():
    calls = []

    def encode(inputs, kind, timing):
        calls.append((inputs, kind))
        return np.array([[value, 1] for value in inputs])

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, batch_size=8, max_tokens=8)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[scheduler.submit(str(i), [i], [2], "document") for i in range(16)]
                ),
                timeout=3,
            )
            assert [result.tolist() for result in results] == [[[i, 1]] for i in range(16)]
            assert [len(inputs) for inputs, _ in calls] == [4, 4, 4, 4]
            stats = scheduler.stats()
            assert stats["known_input_tokens"] == stats["padded_tokens"] == 32
            assert stats["inference_seconds"] > 0
        finally:
            await scheduler.close()


@pytest.mark.asyncio
async def test_small_request_gets_a_turn_before_large_request_finishes():
    started, release = threading.Event(), threading.Event()
    calls = []

    def encode(inputs, kind, timing):
        calls.append(inputs)
        if len(calls) == 1:
            started.set()
            assert release.wait(3)
        return np.array([[value, 1] for value in inputs])

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, batch_size=2, wait_ms=0)
        heavy = asyncio.create_task(scheduler.submit("heavy", list(range(8)), [1] * 8, "document"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            light = asyncio.create_task(scheduler.submit("light", [99], [1], "document"))
            await asyncio.sleep(0)
            release.set()
            assert (await asyncio.wait_for(light, 2)).tolist() == [[99, 1]]
            assert (await asyncio.wait_for(heavy, 2))[:, 0].tolist() == list(range(8))
            assert calls[0] == [0, 1]
            assert calls[1][0] == 99
        finally:
            release.set()
            await scheduler.close()


@pytest.mark.asyncio
async def test_queries_media_and_over_budget_inputs_do_not_merge():
    calls = []

    def encode(inputs, kind, timing):
        calls.append((inputs, kind))
        return np.ones((len(inputs), 2))

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, batch_size=8, max_tokens=8)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("a", [1], [2], "query"),
                    scheduler.submit("b", [2, 3], [None, None], "document"),
                    scheduler.submit("c", [4, 5], [100, 2], "document"),
                ),
                3,
            )
            assert calls == [
                ([1], "query"),
                ([2], "document"),
                ([4], "document"),
                ([3], "document"),
                ([5], "document"),
            ]
        finally:
            await scheduler.close()


@pytest.mark.asyncio
async def test_worker_failure_does_not_stall_later_requests():
    def encode(inputs, kind, timing):
        if 0 in inputs:
            raise ValueError("invalid input")
        return np.ones((len(inputs), 2))

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, wait_ms=0)
        try:
            with pytest.raises(ValueError, match="invalid input"):
                await asyncio.wait_for(scheduler.submit("a", [0], [1], "document"), 2)
            assert (
                await asyncio.wait_for(scheduler.submit("b", [1], [1], "document"), 2)
            ).shape == (1, 2)
        finally:
            await scheduler.close()


@pytest.mark.asyncio
async def test_cancelling_queued_request_removes_its_work():
    started, release = threading.Event(), threading.Event()
    calls = []

    def encode(inputs, kind, timing):
        calls.append(inputs)
        started.set()
        assert release.wait(3)
        return np.ones((len(inputs), 2))

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, wait_ms=0)
        first = asyncio.create_task(scheduler.submit("a", [1], [1], "document"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            queued = asyncio.create_task(scheduler.submit("b", [2], [1], "document"))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            release.set()
            await asyncio.wait_for(first, 2)
            assert calls == [[1]]
            assert not scheduler.ready
        finally:
            release.set()
            await scheduler.close()


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.asyncio
async def test_cancellation_keeps_capacity_reserved_until_thread_finishes(scheduled):
    started, release = threading.Event(), threading.Event()
    admission = InferenceAdmission(1, 1)

    def encode(*args):
        started.set()
        assert release.wait(3)
        return np.ones((1, 2))

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, wait_ms=0)

        async def work():
            with admission.slot("a") as reservation:
                reservation.reserve_tokens(100)
                if scheduled:
                    await scheduler.submit("a", [1, 2], [None, None], "document")
                else:
                    await run_in_threadpool_with_executor(executor, encode)

        task = asyncio.create_task(work())
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            assert admission.stats()["active_requests"] == 1
            assert admission.stats()["active_tokens"] == 100
            with pytest.raises(HTTPException) as exc:
                with admission.slot("b"):
                    pytest.fail("Capacity should still be reserved")
            assert exc.value.status_code == 429
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert admission.stats()["active_requests"] == 0
            assert admission.stats()["active_tokens"] == 0
            assert not scheduler.ready
        finally:
            release.set()
            await scheduler.close()


def test_admission_limits_global_and_per_user_and_releases_on_error():
    admission = InferenceAdmission(2, 1)
    with admission.slot("a"):
        with pytest.raises(HTTPException) as exc:
            with admission.slot("a"):
                pytest.fail("per-user limit")
        assert exc.value.headers == {"Retry-After": "1"}
        with admission.slot("b"):
            with pytest.raises(HTTPException):
                with admission.slot("c"):
                    pytest.fail("global limit")
    with pytest.raises(ValueError):
        with admission.slot("a"):
            raise ValueError()
    assert admission.stats()["active_requests"] == 0
    assert admission.stats()["users"] == {}


@pytest.mark.asyncio
async def test_multiple_workers_overlap_without_reordering_a_request():
    started, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    active, peak = 0, 0

    def encode(inputs, kind, timing):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
        assert release.wait(3)
        with lock:
            active -= 1
        return np.array([[value, 1] for value in inputs])

    with ThreadPoolExecutor(max_workers=2) as executor:
        scheduler = EmbeddingScheduler(executor, encode, workers=2, batch_size=2, wait_ms=0)
        tasks = [
            asyncio.create_task(scheduler.submit(str(i), list(range(6)), [1] * 6, "document"))
            for i in range(2)
        ]
        try:
            assert await asyncio.to_thread(started.wait, 2)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert peak == 2
            assert all(result[:, 0].tolist() == list(range(6)) for result in results)
        finally:
            release.set()
            await scheduler.close()


@pytest.mark.parametrize("blocker_kind,blocker_length", [("query", 4), ("document", 9)])
@pytest.mark.asyncio
async def test_packing_scans_past_an_incompatible_or_nonfitting_tenant(
    blocker_kind, blocker_length
):
    calls = []

    def encode(inputs, kind, timing):
        calls.append((inputs, kind))
        return np.array([[value, 1] for value in inputs])

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, batch_size=8, max_tokens=8, wait_ms=0)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("a", [1], [4], "document"),
                    scheduler.submit("b", [2], [blocker_length], blocker_kind),
                    scheduler.submit("c", [3], [4], "document"),
                ),
                2,
            )
            assert calls == [([1, 3], "document"), ([2], blocker_kind)]
            assert [result.tolist() for result in results] == [[[i, 1]] for i in range(1, 4)]
        finally:
            await scheduler.close()


@pytest.mark.asyncio
async def test_bulk_request_skips_collection_on_all_its_batches():
    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(
            executor,
            lambda inputs, *args: np.ones((len(inputs), 2)),
            batch_size=8,
            wait_ms=1000,
        )
        try:
            # A one-second window makes an accidental collection wait observable
            # without comparing small timing differences on a loaded machine.
            await asyncio.wait_for(scheduler.submit("a", list(range(40)), [1] * 40, "document"), 1)
            assert scheduler.stats()["collection_waits"] == 0
            assert scheduler.stats()["batches"] == 5
        finally:
            await scheduler.close()


def test_token_budget_distinguishes_dense_and_small_requests_and_releases():
    admission = InferenceAdmission(64, 8, max_tokens=100, per_user_tokens=60)
    with admission.slot("a") as a:
        a.reserve_tokens(60)
        with admission.slot("a") as b:
            with pytest.raises(HTTPException):
                b.reserve_tokens(1)
        with admission.slot("b") as b:
            b.reserve_tokens(40)
            with admission.slot("c") as c:
                with pytest.raises(HTTPException):
                    c.reserve_tokens(1)
            assert admission.stats()["active_tokens"] == 100
        assert admission.stats()["active_tokens"] == 60
    assert admission.stats()["active_tokens"] == 0
    assert admission.stats()["user_tokens"] == {}


def test_oversized_request_runs_alone_and_keeps_later_work_out():
    admission = InferenceAdmission(64, 8, max_tokens=100, per_user_tokens=60)
    with admission.slot("a") as a:
        a.reserve_tokens(1000)
        with admission.slot("b") as b:
            with pytest.raises(HTTPException):
                b.reserve_tokens(1)
    with admission.slot("b") as b:
        b.reserve_tokens(1)
    assert admission.stats()["active_tokens"] == 0


def test_retry_after_uses_outstanding_work_and_observed_service_rate():
    admission = InferenceAdmission(1, 1)
    admission.observe(100, 1)
    with admission.slot("a") as a:
        a.reserve_tokens(1700)
        with pytest.raises(HTTPException) as exc:
            with admission.slot("b"):
                pytest.fail("request cap")
        assert exc.value.headers["Retry-After"] == "17"
    assert admission.stats()["active_tokens"] == 0


@pytest.mark.asyncio
async def test_media_later_in_a_request_does_not_block_packing_its_text_prefix():
    calls = []

    def encode(inputs, kind, timing):
        calls.append(inputs)
        return np.array([[value, 1] for value in inputs])

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, batch_size=8, max_tokens=8, wait_ms=0)
        try:
            first, second = await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("a", [1, 2], [4, None], "document"),
                    scheduler.submit("b", [3], [4], "document"),
                ),
                2,
            )
            assert calls == [[1, 3], [2]]
            assert first[:, 0].tolist() == [1, 2]
            assert second[:, 0].tolist() == [3]
        finally:
            await scheduler.close()


def test_warmed_small_request_retry_has_no_fixed_seven_second_floor():
    from contextlib import ExitStack

    admission = InferenceAdmission(64, 8)
    with ExitStack() as stack:
        for i in range(64):
            reservation = stack.enter_context(admission.slot(str(i)))
            reservation.reserve_tokens(60)
        # First actual batch seeds both work and request throughput.
        admission.observe(480, 0.024, requests=8)
        with pytest.raises(HTTPException) as exc:
            with admission.slot("overflow"):
                pytest.fail("request cap")
        assert exc.value.headers["Retry-After"] == "1"
        assert admission.stats()["estimated_requests_per_second"] > 300


@pytest.mark.asyncio
async def test_staggered_singles_collect_until_full_or_quiet():
    calls = []

    def encode(inputs, kind, timing):
        calls.append(inputs)
        return np.ones((len(inputs), 2))

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, encode, wait_ms=500, quiet_ms=100)
        tasks = []
        try:
            # Gaps are shorter than the quiet period; filling the batch wakes
            # the worker without paying the full half-second collection cap.
            for i in range(8):
                tasks.append(asyncio.create_task(scheduler.submit(str(i), [i], [1], "document")))
                await asyncio.sleep(0.01)
            await asyncio.wait_for(asyncio.gather(*tasks), 0.3)
            assert calls == [list(range(8))]
        finally:
            await scheduler.close()


@pytest.mark.asyncio
async def test_partial_collection_flushes_on_quiet_without_waiting_for_cap():
    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(
            executor,
            lambda inputs, *args: np.ones((len(inputs), 2)),
            wait_ms=1000,
            quiet_ms=10,
        )
        try:
            await asyncio.wait_for(scheduler.submit("a", [1], [1], "document"), 0.5)
            assert scheduler.stats()["batches"] == 1
            assert scheduler.stats()["collection_waits"] == 1
        finally:
            await scheduler.close()


@pytest.mark.asyncio
async def test_collection_cap_cannot_be_extended_by_continuous_arrivals():
    encoded = asyncio.Event()

    with ThreadPoolExecutor(max_workers=1) as executor:
        loop = asyncio.get_running_loop()

        def encode(inputs, *args):
            loop.call_soon_threadsafe(encoded.set)
            return np.ones((len(inputs), 2))

        scheduler = EmbeddingScheduler(executor, encode, batch_size=1000, wait_ms=50, quiet_ms=1000)
        tasks = []
        stop = asyncio.Event()

        async def produce():
            for i in range(30):
                if stop.is_set():
                    return
                tasks.append(asyncio.create_task(scheduler.submit(str(i), [i], [1], "document")))
                await asyncio.sleep(0.01)

        producer = asyncio.create_task(produce())
        try:
            # The cap must flush while arrivals are still extending quiet time.
            await asyncio.wait_for(encoded.wait(), 0.5)
            assert not producer.done()
        finally:
            stop.set()
            await producer
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
            await scheduler.close()


@pytest.mark.asyncio
async def test_admission_wait_is_bounded_by_request_slots_and_wakes_on_release():
    admission = InferenceAdmission(2, 1, max_tokens=1)
    reserved = asyncio.Event()
    release = asyncio.Event()

    async def busy():
        with admission.slot("busy") as reservation:
            reservation.reserve_tokens(1)
            reserved.set()
            await release.wait()

    async def queued():
        with admission.slot("queued") as reservation:
            await reservation.wait_for_tokens(1, 1)
            assert admission.stats()["active_tokens"] == 1

    busy_task = asyncio.create_task(busy())
    await reserved.wait()
    queued_task = asyncio.create_task(queued())
    await asyncio.sleep(0)
    assert admission.stats()["waiting_requests"] == 1
    with pytest.raises(HTTPException):
        with admission.slot("overflow"):
            pytest.fail("waiting must not bypass request limits")
    release.set()
    await asyncio.wait_for(asyncio.gather(busy_task, queued_task), 1)
    assert admission.stats()["active_requests"] == 0
    assert admission.stats()["active_tokens"] == 0
    assert admission.stats()["waiting_requests"] == 0


@pytest.mark.asyncio
async def test_admission_wait_times_out_and_cancellation_removes_waiter():
    admission = InferenceAdmission(3, 1, max_tokens=1)

    async def queued(username, timeout):
        with admission.slot(username) as reservation:
            await reservation.wait_for_tokens(1, timeout)

    with admission.slot("busy") as busy:
        busy.reserve_tokens(1)
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(queued("timeout", 0.01), 1)
        assert exc.value.status_code == 429
        task = asyncio.create_task(queued("cancel", 1))
        await asyncio.sleep(0)
        assert admission.stats()["waiting_requests"] == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert admission.stats()["waiting_requests"] == 0
        assert admission.stats()["active_requests"] == 1
        assert admission.stats()["active_tokens"] == 1
    assert admission.stats()["active_tokens"] == 0


def test_cold_small_request_burst_does_not_inherit_a_fixed_depth_delay():
    from contextlib import ExitStack

    admission = InferenceAdmission(64, 8)
    with ExitStack() as stack:
        for i in range(64):
            reservation = stack.enter_context(admission.slot(str(i)))
            reservation.reserve_tokens(60)
        with pytest.raises(HTTPException) as exc:
            with admission.slot("overflow"):
                pytest.fail("request cap")
        assert exc.value.headers["Retry-After"] == "1"


@pytest.mark.asyncio
async def test_rate_samples_count_a_split_request_once_across_its_batches():
    observations = []

    def observe(tokens, duration, requests):
        observations.append(requests)

    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(
            executor,
            lambda inputs, *args: np.ones((len(inputs), 2)),
            wait_ms=0,
            observe=observe,
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("small", [1], [1], "document"),
                    scheduler.submit("large", list(range(16)), [1] * 16, "document"),
                ),
                1,
            )
            assert sum(observations) == 2
            assert scheduler.stats()["batches"] == 3
        finally:
            await scheduler.close()


@pytest.mark.asyncio
async def test_default_collection_tolerates_three_ms_arrival_gaps_and_exposes_effective_settings(
    monkeypatch,
):
    from types import SimpleNamespace
    import bananabread.scheduling as scheduling
    from bananabread.utils import ExecutionTiming

    now = [0.0]
    monkeypatch.setattr(scheduling, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    with ThreadPoolExecutor(max_workers=1) as executor:
        scheduler = EmbeddingScheduler(executor, lambda inputs, *args: np.ones((len(inputs), 2)))
        try:
            loop = asyncio.get_running_loop()
            for i in range(8):
                now[0] = i * 0.003
                request = scheduling._Request(
                    str(i),
                    [i],
                    [1],
                    "document",
                    loop.create_future(),
                    ExecutionTiming(),
                    submitted=now[0],
                )
                scheduler._enqueue(request)
                delay = scheduler._collection_delay()
                if i < 7:
                    assert delay == pytest.approx(min(0.005, 0.025 - now[0]))
                else:
                    assert delay == 0  # Full at 21 ms, before the absolute cap.
            assert scheduler.stats()["wait_ms"] == 25
            assert scheduler.stats()["quiet_ms"] == 5
        finally:
            await scheduler.close()
