"""Auth v2 (P2, docs/plan-auth-v2.md): Geräte-Cookies + Login-Bremse."""
import time

from semanticswap.auth import DeviceCookieSigner, LoginBrake


def test_cookie_roundtrip_and_role():
    signer = DeviceCookieSigner(b"s" * 32)
    token = signer.issue("family", days=90)
    assert signer.verify(token) == "family"
    assert signer.issue("admin") != token  # Nonce -> Tokens sind einmalig


def test_cookie_tampering_is_rejected():
    signer = DeviceCookieSigner(b"s" * 32)
    token = signer.issue("family")
    payload, sig = token.rsplit(".", 1)
    assert signer.verify(payload + "." + "0" * len(sig)) is None
    assert signer.verify("kaputt") is None
    # anderes Server-Secret (Rotation) = globaler Logout
    assert DeviceCookieSigner(b"x" * 32).verify(token) is None


def test_cookie_expiry():
    signer = DeviceCookieSigner(b"s" * 32)
    token = signer.issue("family", days=0)  # sofort abgelaufen
    assert signer.verify(token) is None


def test_login_brake_delays_after_failures():
    brake = LoginBrake(free_attempts=3, base_delay=10.0)
    ip = "203.0.113.7"
    for _ in range(3):
        brake.register_failure(ip)
        allowed, wait = brake.check(ip)
        assert allowed and wait == 0
    brake.register_failure(ip)  # 4. Fehlversuch -> Bremse greift
    allowed, wait = brake.check(ip)
    assert not allowed and wait > 0
    # andere IP bleibt unbehelligt
    assert brake.check("198.51.100.1") == (True, 0)


def test_login_brake_resets_on_success():
    brake = LoginBrake(free_attempts=1, base_delay=10.0)
    ip = "203.0.113.8"
    brake.register_failure(ip)
    brake.register_failure(ip)
    assert not brake.check(ip)[0]
    brake.register_success(ip)
    assert brake.check(ip) == (True, 0)


def test_login_brake_expires_by_time(monkeypatch):
    brake = LoginBrake(free_attempts=1, base_delay=5.0)
    ip = "203.0.113.9"
    brake.register_failure(ip)
    brake.register_failure(ip)
    assert not brake.check(ip)[0]
    monkeypatch.setattr(time, "monotonic", lambda: time.time() + 3600)
    assert brake.check(ip)[0]
