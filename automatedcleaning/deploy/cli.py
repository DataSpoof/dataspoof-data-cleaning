"""Command line: ``automatedcleaning serve | score | docker | export | monitor``.

Also available as ``python -m automatedcleaning.deploy ...``.
"""

from __future__ import annotations

import argparse
import json
import sys


def _model(args):
    if getattr(args, "registry", None):
        from ..registry import ModelRegistry
        name, _, ver = args.model.partition(":")
        reg = ModelRegistry(args.registry)
        if ver.isdigit():
            return reg.get(name, version=int(ver))
        return reg.get(name, stage=ver or None)
    return None


def _model_path(args):
    mv = _model(args)
    return str(mv.model_path) if mv is not None else args.model


def main(argv=None):
    p = argparse.ArgumentParser(prog="automatedcleaning", description="Serve, score and package models.")
    p.add_argument("--registry", help="registry directory; then --model is NAME, NAME:VERSION or NAME:STAGE")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve a model over REST (development server)")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-rows", type=int, default=10_000)
    s.add_argument("--prediction-log")

    b = sub.add_parser("score", help="batch-score a CSV/Parquet/NDJSON file")
    b.add_argument("--model", required=True)
    b.add_argument("--input", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--chunk-size", type=int, default=100_000)
    b.add_argument("--keep", default="all", help="'all', 'none', or comma-separated id columns")

    d = sub.add_parser("docker", help="write a Docker build context for the model")
    d.add_argument("--model", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--name", default="model")
    d.add_argument("--version", default="1")
    d.add_argument("--port", type=int, default=8000)

    e = sub.add_parser("export", help="export a model (bundle, pickle or onnx)")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--format", default="bundle", choices=["bundle", "pickle", "onnx"])

    m = sub.add_parser("monitor", help="check a batch of data against a saved monitor")
    m.add_argument("--model", required=True)
    m.add_argument("--monitor", required=True, help="monitor JSON saved with Monitor.save()")
    m.add_argument("--input", required=True)

    args = p.parse_args(argv)
    path = _model_path(args)
    if args.cmd == "serve":
        from .server import serve
        serve(path, host=args.host, port=args.port, max_rows=args.max_rows, prediction_log=args.prediction_log)
    elif args.cmd == "score":
        from .scoring import batch_score
        keep = "all" if args.keep == "all" else (None if args.keep == "none" else args.keep.split(","))
        print(json.dumps(batch_score(path, args.input, args.output, chunk_size=args.chunk_size, keep=keep),
                         indent=2, default=str))
    elif args.cmd == "docker":
        from .export import build_docker
        out = build_docker(path, args.out, name=args.name, version=args.version, port=args.port)
        print(f"Docker context written to {out}. Build it with: docker build -t {args.name} {out}")
    elif args.cmd == "export":
        from .export import export_model
        print(export_model(path, args.out, format=args.format))
    elif args.cmd == "monitor":
        import polars as pl

        from ..monitoring import Monitor
        from .scoring import _batches, resolve_model
        from pathlib import Path
        monitor = Monitor.load(args.monitor, resolve_model(path))
        batch = pl.concat(list(_batches(Path(args.input), 1_000_000)), how="vertical_relaxed")
        report = monitor.check(batch)
        print(report.summary())
        return 1 if report.alerts else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
