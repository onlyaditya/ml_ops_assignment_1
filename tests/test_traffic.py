"""Traffic generator: payloads must be valid API JSON; network is mocked."""
import json

from src import traffic
from src.config import FEATURES


def test_patients_are_json_ready():
    patients = traffic.load_patients()
    assert len(patients) == 303
    first = patients[0]
    assert list(first) == FEATURES
    assert isinstance(first["cp"], int)            # codes sent as ints, not 4.0
    json.dumps(patients, allow_nan=False)          # no NaN sneaks into the JSON
    assert any(p["ca"] is None for p in patients)  # real missing values kept as null


def test_main_sends_requests_and_some_invalid(monkeypatch, capsys):
    sent = []

    def fake_send(url, payload):
        sent.append(payload)
        return 200

    monkeypatch.setattr(traffic, "send", fake_send)
    monkeypatch.setattr(traffic.time, "sleep", lambda s: None)
    # stop after a fixed number of loop iterations instead of wall-clock time
    ticks = iter(range(1000))
    monkeypatch.setattr(traffic.time, "monotonic", lambda: next(ticks) * 0.01)

    traffic.main(["--duration", "2", "--bad-share", "0.5", "--seed", "1"])

    assert 150 <= len(sent) <= 200
    broken = [p for p in sent if any(p.get(f) == v for f, v in traffic.BAD_INPUTS)]
    assert 0.3 < len(broken) / len(sent) < 0.7
    assert "Done:" in capsys.readouterr().out
