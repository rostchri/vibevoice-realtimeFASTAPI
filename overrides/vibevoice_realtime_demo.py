import argparse
import os

import uvicorn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--model_path", type=str, default="microsoft/VibeVoice-Realtime-0.5B")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mpx", "mps"])
    p.add_argument("--reload", action="store_true", help="Reload the model or not")
    p.add_argument("--inference_steps", type=int, default=5)
    p.add_argument(
        "--lazy-load",
        action="store_true",
        help="Defer model initialization until the first speech request",
    )
    p.add_argument(
        "--startup-warmup",
        dest="startup_warmup",
        action="store_true",
        default=None,
        help="Warm the model during startup",
    )
    p.add_argument(
        "--no-startup-warmup",
        dest="startup_warmup",
        action="store_false",
        help="Skip startup warmup",
    )
    args = p.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MODEL_DEVICE"] = args.device
    os.environ["INFERENCE_STEPS"] = str(args.inference_steps)
    if args.lazy_load:
        os.environ["ENABLE_LAZY_LOAD"] = "true"
    if args.startup_warmup is not None:
        os.environ["ENABLE_STARTUP_WARMUP"] = "true" if args.startup_warmup else "false"
    elif args.lazy_load:
        os.environ["ENABLE_STARTUP_WARMUP"] = "false"

    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
