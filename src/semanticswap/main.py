"""Headless-Einstiegspunkt (ADR-006): python -m semanticswap.main [config.yaml]"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import load_config
from .gateway import create_app


def build_app(config_path: str | None = None):
    path = Path(config_path or "config.yaml")
    cfg = load_config(path) if path.exists() else None
    if cfg is None:
        from .config import AppConfig

        logging.warning("Keine config.yaml gefunden - Defaults werden verwendet")
        cfg = AppConfig()
    return create_app(cfg), cfg


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    app, cfg = build_app(config_path)

    import uvicorn

    # proxy_headers=False ist SICHERHEITSRELEVANT: sonst überschreibt uvicorn
    # request.client.host aus X-Forwarded-For (vom Funnel/127.0.0.1 als trusted
    # forwarder akzeptiert). Die Tailnet-Vertrauensprüfung MUSS auf der echten
    # TCP-Quelle beruhen — sonst genügt ein gefälschter Header, um als Ivos
    # Gerät durchzugehen. Das Logbuch liest X-Forwarded-For selbst (nur Anzeige).
    uvicorn.run(app, host=cfg.gateway.host, port=cfg.gateway.port,
                proxy_headers=False, forwarded_allow_ips=[])


if __name__ == "__main__":
    run()
