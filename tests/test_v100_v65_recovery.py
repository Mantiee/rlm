import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rlm.v100 import (
    backtest_audit,
    backtest_learning,
    backtesting,
    compute_worker,
    remote_connect,
    resource_budget,
)
from tests.test_v100_backtesting import candles

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"


def stored_report(root):
    return backtesting.run(
        root,
        "A",
        {"price_url": URL, "event_url": "", "rule": "momentum", "fee_bps": 50, "slippage_bps": 10},
        source_snapshot=(URL, json.dumps(candles())),
    )


def test_audit_replays_equity_and_quarantines_changed_source(tmp_path):
    from pathlib import Path

    path = Path(stored_report(tmp_path)["report"])
    first = backtest_audit.run(tmp_path)
    row = first["reports"][0]
    assert row["state"] == "reproduced" and row["postmortem_training_admitted"]
    assert not row["forecast_training_admitted"]
    source = path.parent / "source-0.json"
    source.write_text("[]")
    before = path.read_bytes()
    second = backtest_audit.run(tmp_path)["reports"][0]
    assert second["state"] == "quarantined" and "changed" in second["reason"]
    assert path.read_bytes() == before and backtest_learning.records(tmp_path) == []


@pytest.mark.parametrize("field,value", [(1, 1000), (2, 1), (5, -1), (2, float("inf"))])
def test_bad_ohlcv_is_rejected(field, value):
    rows = candles()
    rows[50][field] = value
    with pytest.raises(ValueError, match="OHLCV"):
        backtesting.bars_from_source(URL, json.dumps(rows), datetime.now(UTC))


def test_gapped_source_cannot_be_postmortem_training(tmp_path):
    from pathlib import Path

    rows = candles()
    rows = rows[:50] + rows[51:]
    report = backtesting.run(
        tmp_path,
        "A",
        {"price_url": URL, "event_url": "", "rule": "momentum", "fee_bps": 50, "slippage_bps": 10},
        source_snapshot=(URL, json.dumps(rows)),
    )
    assert report["source_quality"]["gaps"] == 1
    assert backtest_audit.inspect_report(Path(report["report"]))["state"] == "quarantined"


def test_negative_return_is_not_a_glitch_label(tmp_path):
    from pathlib import Path

    path = Path(stored_report(tmp_path)["report"])
    labels = json.loads(backtest_learning.verified(path)["messages"][-1]["content"])
    assert labels["negative_return_proves_data_corruption"] is False
    assert labels["drawdown_is_standard_deviation"] is False
    altered = copy.deepcopy(json.loads(path.read_text()))
    altered["double_cost_stress"]["max_sampled_drawdown"] = altered["double_cost_stress"][
        "net_return"
    ]
    path.write_text(json.dumps(altered))
    assert backtest_audit.inspect_report(path)["state"] == "quarantined"


@pytest.mark.parametrize(
    "cpu,ram,slots", [(10, 12, 2), (70, 12, 1), (90, 12, 0), (10, 1, 0), (10, 3, 1)]
)
def test_local_drones_admit_only_measured_headroom(monkeypatch, cpu, ram, slots):
    import psutil

    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: cpu)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=ram * 2**30))
    assert resource_budget.local_budget()["cpu_slots"] == slots


def test_protected_foreground_pauses_worker_instead_of_crashing(monkeypatch):
    import psutil

    monkeypatch.setattr(compute_worker.sys, "platform", "win32")
    user = SimpleNamespace(
        GetForegroundWindow=Mock(return_value=1), GetWindowThreadProcessId=Mock()
    )
    monkeypatch.setattr(compute_worker.ctypes, "WinDLL", lambda *a, **k: user, raising=False)
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.AccessDenied(1)))
    assert compute_worker.foreground_busy() is True


@pytest.mark.parametrize("cpu,slots", [(10, 2), (30, 2), (45, 1)])
def test_windows_child_affinity_adapts_within_two_slots(monkeypatch, cpu, slots):
    import psutil

    monkeypatch.setattr(compute_worker.sys, "platform", "win32")
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval: cpu)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=12 * 2**30))
    child = SimpleNamespace(cpu_affinity=Mock())
    host = SimpleNamespace(cpu_affinity=lambda: list(range(16)))
    monkeypatch.setattr(psutil, "Process", lambda pid: child if pid == -1 else host)
    assert compute_worker.adaptive_child_budget(-1)["cpu_affinity_slots"] == slots
    child.cpu_affinity.assert_called_once_with(list(range(16))[-slots:])


def test_tailscale_connection_uses_actual_dns_and_requires_login(monkeypatch):
    monkeypatch.setattr(remote_connect.getpass, "getuser", lambda: "marek")
    result = remote_connect.links(
        {"BackendState": "Running", "Self": {"DNSName": "debian1.example.ts.net."}}
    )
    assert result["dashboard"] == "https://debian1.example.ts.net"
    assert result["chat"].endswith("/chat")
    assert result["ssh"] == "ssh marek@debian1.example.ts.net"
    with pytest.raises(ValueError, match="not connected"):
        remote_connect.links({"BackendState": "NeedsLogin"})


def test_transient_mailbox_failure_retries_with_visible_reason(tmp_path, monkeypatch, capsys):
    calls = []

    def flaky(*args):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("SMB disconnected")

    monkeypatch.setattr(compute_worker, "service", flaky)
    monkeypatch.setattr(compute_worker.time, "sleep", lambda seconds: None)
    compute_worker.recover_service(tmp_path, "windows-cpu", tmp_path / "kernel.py")
    assert len(calls) == 2 and "SMB disconnected" in capsys.readouterr().out


def test_drawdown_uses_peak_not_standard_deviation():
    bars = [
        {"open_time": i, "available_at": i + 1, "open": price, "close": price}
        for i, price in enumerate([100, 100, 120, 90, 110])
    ]
    result = backtesting.simulate(bars, 2, 5, "buy_hold", 1, 0, 0)
    assert result["net_return"] == pytest.approx(110 / 120 - 1)
    assert result["max_sampled_drawdown"] == pytest.approx(1 - 90 / 120)
