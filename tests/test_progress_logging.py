import logging

import bananabread.utils as bb_utils
from bananabread.utils import route_tqdm_to_logger
from tqdm.auto import tqdm


def _fast_logs(monkeypatch):
    route_tqdm_to_logger()
    monkeypatch.setattr(bb_utils, "TQDM_LOG_INTERVAL_SECONDS", 0.0)


def test_enabled_bar_logs_throttled_lines_and_completion(caplog, monkeypatch):
    _fast_logs(monkeypatch)
    with caplog.at_level(logging.INFO, logger="BananaBread-Emb"):
        bar = tqdm(total=10, desc="model.safetensors")
        bar.update(4)
        bar.update(6)
        bar.close()

    lines = [r.message for r in caplog.records if "model.safetensors" in r.message]
    assert any("4/10" in line and "40%" in line for line in lines)
    assert any("10/10" in line and "100%" in line for line in lines)
    assert any(line.startswith("✅") and "in " in line for line in lines)


def test_bar_renders_no_ansi_output(caplog, capsys, monkeypatch):
    _fast_logs(monkeypatch)
    with caplog.at_level(logging.INFO, logger="BananaBread-Emb"):
        bar = tqdm(total=3, desc="Fetching 3 files")
        for _ in range(3):
            bar.update(1)
        bar.close()

    captured = capsys.readouterr()
    assert "\r" not in captured.err  # no rendered bar carriage returns
    assert "%|" not in captured.err


def test_disabled_bar_stays_silent(caplog, monkeypatch):
    _fast_logs(monkeypatch)
    with caplog.at_level(logging.INFO, logger="BananaBread-Emb"):
        bar = tqdm(total=5, desc="silent-op", disable=True)
        bar.update(5)
        bar.close()

    assert not [r for r in caplog.records if "silent-op" in r.message]


def test_instant_bar_emits_nothing(caplog, monkeypatch):
    # Cached snapshots open and close bars immediately; keep them quiet.
    route_tqdm_to_logger()
    monkeypatch.setattr(bb_utils, "TQDM_LOG_INTERVAL_SECONDS", 999.0)
    with caplog.at_level(logging.INFO, logger="BananaBread-Emb"):
        bar = tqdm(total=15, desc="Fetching 15 files")
        bar.update(15)
        bar.close()

    assert not [r for r in caplog.records if "Fetching 15 files" in r.message]


def test_hub_tqdm_subclass_is_routed(caplog, monkeypatch):
    from huggingface_hub.utils import tqdm as hub_tqdm

    _fast_logs(monkeypatch)
    with caplog.at_level(logging.INFO, logger="BananaBread-Emb"):
        bar = hub_tqdm(total=2, desc="Fetching 2 files")
        bar.update(2)
        bar.close()

    lines = [r.message for r in caplog.records if "Fetching 2 files" in r.message]
    assert any("2/2" in line for line in lines)


def test_indeterminate_bar_reports_count_and_rate(caplog, monkeypatch):
    _fast_logs(monkeypatch)
    with caplog.at_level(logging.INFO, logger="BananaBread-Emb"):
        bar = tqdm(desc="loading weights")
        bar.update(146)
        bar.close()

    lines = [r.message for r in caplog.records if "loading weights" in r.message]
    assert any("146" in line and "/s" in line for line in lines)
