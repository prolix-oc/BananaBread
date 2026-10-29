"""Idle GPU cleanup must never evict allocator blocks during live inference."""

from types import SimpleNamespace

import pytest

import bananabread.cache as cache_module
from bananabread.cache import CUDACacheManager


@pytest.mark.parametrize("fail", [False, True])
def test_idle_cleanup_waits_for_all_workers_and_restarts_ttl_on_completion(monkeypatch, fail):
    now = [0.0]
    monkeypatch.setattr(cache_module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(cache_module.torch.cuda, "is_available", lambda: False)
    manager = CUDACacheManager(ttl_seconds=10, min_clear_interval=1)
    manager.cuda_available = True
    monkeypatch.setattr(manager, "get_cuda_memory_stats", lambda: {"reserved_pct": 90})
    now[0] = 20
    assert manager.should_clear_cache()
    with manager.inference_activity():
        try:
            with manager.inference_activity():
                now[0] = 100  # Longer than TTL, with overlapping workers.
                assert manager.active_inferences == 2
                assert not manager.should_clear_cache()
                if fail:
                    raise ValueError("inference failed")
        except ValueError:
            assert fail
        assert manager.active_inferences == 1
        assert not manager.should_clear_cache()
        now[0] = 200
    assert manager.active_inferences == 0
    assert manager.last_inference_time == 200
    assert not manager.should_clear_cache()
    now[0] = 211
    assert manager.should_clear_cache()
