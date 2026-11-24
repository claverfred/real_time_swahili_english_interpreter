"""``serve``: local web server with the browser UI, live WebSocket and file upload.

    python -m src.cli serve --profile cpu                    # http://127.0.0.1:8000
    python -m src.cli serve --profile cpu --open             # ... and open the browser
    $env:SWI_SERVER_TOKEN = "..."; python -m src.cli serve --host 0.0.0.0   # network: token required

The token is read from the environment (``server.token_env``), never from the command line, so
it does not end up in shell history or process listings. If a token is required and none is set,
one is generated and the access link is printed once.
"""

from __future__ import annotations

import argparse
import os
import sys

from ..config import load_config


def add_parser(sub: argparse._SubParsersAction, add_profile_args) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("serve", help="web UI + WebSocket/HTTP API (local by default)")
    p.add_argument("--host", help="bind address (default server.host = 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default server.port = 8000)")
    p.add_argument("--no-preload", action="store_true", help="load models on the first session instead of at startup")
    p.add_argument("--open", action="store_true", help="open the UI in the default browser")
    add_profile_args(p)
    p.set_defaults(func=cmd_serve)


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from ..server import AccessPolicy, create_app, is_loopback
    except ImportError as exc:
        print(f"serve needs the ui extra: pip install -e .[ui]  ({exc})", file=sys.stderr)
        return 2
    overrides = list(args.overrides)
    if args.host:
        overrides.append(f"server.host={args.host}")
    if args.port:
        overrides.append(f"server.port={args.port}")
    if args.no_preload:
        overrides.append("server.preload=false")
    cfg = load_config(args.profile, overrides)
    s = cfg.server
    policy, generated = AccessPolicy.for_host(
        s.host, os.environ.get(s.token_env) or None, require_token=s.require_token,
        allowed_origins=s.allowed_origins, rate_limit_per_min=s.rate_limit_per_min)
    if not is_loopback(s.host):
        print(f"warning: listening on {s.host} — reachable from the network. Every request needs the token; "
              "traffic is not encrypted (use an SSH tunnel or a TLS proxy). Browsers only allow the "
              "microphone on https:// or localhost.", file=sys.stderr)
    shown = "127.0.0.1" if s.host in ("0.0.0.0", "::") else s.host
    url = f"http://{shown}:{s.port}/"
    if policy.required:
        if generated:
            print(f"No {s.token_env} set; generated a token for this run.", file=sys.stderr)
        link = f"{url}?token={policy.token}" if generated else f"{url}?token=<value of {s.token_env}>"
    else:
        link = url
    app = create_app(cfg, policy, user_set_silence=any(o.startswith("vad.min_silence_ms=") for o in overrides))
    print(f"Swahili->English interpreter (profile {cfg.profile}) — open {link}", file=sys.stderr)
    print("Models load in the background" if s.preload else "Models load on first use", "; Ctrl+C to stop.",
          file=sys.stderr)
    if args.open:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"{url}?token={policy.token}" if policy.required else url)).start()
    uvicorn.run(app, host=s.host, port=s.port, ws_max_size=s.max_ws_message_bytes, log_level="warning")
    return 0
