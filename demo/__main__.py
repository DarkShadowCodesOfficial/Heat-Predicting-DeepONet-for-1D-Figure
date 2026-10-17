"""Launch the demo:  python -m demo [--preset quick|notebook] [--port 8000]"""
import argparse
import threading
import webbrowser

import uvicorn

from heat_operator.config import PRESETS
from demo.server import create_app


def main():
    ap = argparse.ArgumentParser(prog="python -m demo", description=__doc__)
    ap.add_argument("--preset", default="quick", choices=sorted(PRESETS),
                    help="training budget (quick ~ minutes, notebook = exact notebook settings)")
    ap.add_argument("--retrain", action="store_true", help="ignore any saved checkpoint and train again")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="where trained weights are cached")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    app = create_app(args.preset, args.ckpt_dir, args.retrain)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Demo running at {url}  (preset: {args.preset})\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
