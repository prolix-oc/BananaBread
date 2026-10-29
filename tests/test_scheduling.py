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
