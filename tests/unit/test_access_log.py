"""Auth v2: durables Zugriffs-Logbuch für Zugriffe außerhalb des Tailnets."""
from semanticswap.auth import AccessLog


def test_records_and_tails_newest_first(tmp_path):
    log = AccessLog(tmp_path / "a.jsonl", keep=3)
    for i in range(5):
        log.record(ip=f"1.2.3.{i}", via="edge", method="GET",
                   path="/v1/models", outcome="blocked_401", role=None)
    recent = log.recent()
    assert len(recent) == 3  # Ringpuffer deckelt
    assert recent[0]["ip"] == "1.2.3.4"  # neueste zuerst
    assert "ts" in recent[0]


def test_persists_across_reload(tmp_path):
    p = tmp_path / "a.jsonl"
    AccessLog(p).record(ip="9.9.9.9", via="funnel", method="GET", path="/",
                        outcome="blocked_401", role=None)
    reopened = AccessLog(p)  # nach Neustart: Historie wieder da
    assert reopened.recent()[0]["ip"] == "9.9.9.9"


def test_memory_only_without_path():
    log = AccessLog(None)
    log.record(ip="1.1.1.1", via="edge", method="GET", path="/",
               outcome="blocked_401", role=None)
    assert log.recent()[0]["ip"] == "1.1.1.1"


def test_rotates_when_file_grows(tmp_path):
    p = tmp_path / "a.jsonl"
    log = AccessLog(p, max_bytes=200)
    for i in range(50):
        log.record(ip=f"5.5.5.{i}", via="funnel", method="GET",
                   path="/probe", outcome="blocked_401", role=None)
    assert p.with_suffix(".jsonl.1").exists()  # Rotation hat gegriffen
    assert p.stat().st_size < 4000  # aktive Datei bleibt begrenzt
