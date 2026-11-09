"""Command line: ``mlpilot serve | score | docker | export | monitor | hub`` and the
model CI/CD steps ``train | gate | register | promote | release | latest | deploy-sagemaker | cicd init``.

Also available as ``python -m mlpilot.deploy ...``.
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


def container_command() -> list:
    """The gunicorn command line for the current environment (Vertex sets AIP_HTTP_PORT)."""
    import os
    port = os.environ.get("AIP_HTTP_PORT") or os.environ.get("AC_PORT") or os.environ.get("PORT") or "8080"
    return ["gunicorn", "--bind", f"0.0.0.0:{port}", "--workers", os.environ.get("AC_WORKERS", "2"),
            "--timeout", "120", "--access-logfile", "-", "mlpilot.deploy.wsgi:app"]


def _container():
    import os
    cmd = container_command()
    os.execvp(cmd[0], cmd)              # replaces this process; gunicorn handles signals


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlpilot", description="Serve, score and package models.")
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
    d.add_argument("--port", type=int, default=None)
    d.add_argument("--target", default="generic", choices=["generic", "sagemaker", "vertex", "azureml"])

    c = sub.add_parser("container", help="run the production server inside a container (gunicorn)")
    c.add_argument("extra", nargs="*", help="ignored (SageMaker passes 'serve')")

    e = sub.add_parser("export", help="export a model (bundle, pickle or onnx)")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--format", default="bundle", choices=["bundle", "pickle", "onnx", "skops"])

    m = sub.add_parser("monitor", help="check a batch of data against a saved monitor")
    m.add_argument("--model", required=True)
    m.add_argument("--monitor", required=True, help="monitor JSON saved with Monitor.save()")
    m.add_argument("--input", required=True)

    h = sub.add_parser("hub", help="run / administer the MLPilot Hub server")
    hsub = h.add_subparsers(dest="hub_cmd", required=True)
    hs = hsub.add_parser("serve", help="start the Hub (uvicorn)")
    hs.add_argument("--host", default="127.0.0.1")
    hs.add_argument("--port", type=int, default=8080)
    hs.add_argument("--workers", type=int, default=1)
    ha = hsub.add_parser("create-admin", help="create a platform admin and print an API token (once)")
    ha.add_argument("--name", default="admin")
    ha.add_argument("--email")

    _pipeline_parsers(sub)
    args = p.parse_args(argv)
    if args.cmd in _PIPELINE:
        return _run_pipeline(args)
    if args.cmd == "container":
        return _container()
    if args.cmd == "hub":
        if args.hub_cmd == "create-admin":
            from ..hub.app import bootstrap_admin
            print(bootstrap_admin(name=args.name, email=args.email))
            return 0
        import uvicorn
        uvicorn.run("mlpilot.hub.app:app", host=args.host, port=args.port, workers=args.workers,
                    proxy_headers=True)
        return 0
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
        out = build_docker(path, args.out, name=args.name, version=args.version, port=args.port, target=args.target)
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


# ── model CI/CD commands (see mlpilot.cicd) ───────────────────────
_PIPELINE = ("train", "gate", "register", "promote", "release", "latest", "deploy-sagemaker", "cicd")


def _pipeline_parsers(sub):
    S = argparse.SUPPRESS                    # don't clobber the top-level --registry
    t = sub.add_parser("train", help="split, check, fit AutoML, evaluate and save a model (CI step)")
    t.add_argument("--data", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--out", required=True, help="model directory to write")
    t.add_argument("--test", help="separate test file (default: split --data)")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--group", help="split by this column (no group on both sides)")
    t.add_argument("--time", help="split chronologically by this column")
    t.add_argument("--time-budget", type=float, default=300)
    t.add_argument("--include", help="comma-separated model families")
    t.add_argument("--max-suspicious-share", type=float, help="fail if more training rows look poisoned")
    t.add_argument("--report")

    g = sub.add_parser("gate", help="fail (exit 1) unless the model meets gates.yml on held-out data")
    g.add_argument("--model", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--gates", required=True)
    g.add_argument("--report")

    r = sub.add_parser("register", help="add a model directory to a registry")
    r.add_argument("--model", required=True)
    r.add_argument("--registry", default=S, required=False)
    r.add_argument("--name", required=True)
    r.add_argument("--metrics", help="metrics JSON (e.g. the metrics.json `train` writes)")
    r.add_argument("--stage", default="dev")
    r.add_argument("--description")
    r.add_argument("--sign-key", help="Ed25519 private key to sign the stored model")
    r.add_argument("--report")

    pr = sub.add_parser("promote", help="move a registered version to a stage")
    pr.add_argument("--registry", default=S)
    pr.add_argument("--name", required=True)
    pr.add_argument("--version", type=int, required=True)
    pr.add_argument("--stage", required=True)
    pr.add_argument("--reason")

    rel = sub.add_parser("release", help="release a version to the next environment (environments.yml)")
    rel.add_argument("--config", required=True)
    rel.add_argument("--name", required=True)
    rel.add_argument("--version", type=int, required=True)
    rel.add_argument("--from", dest="frm", required=True)
    rel.add_argument("--to", required=True)
    rel.add_argument("--released-by")
    rel.add_argument("--approved-by", default="", help="comma-separated approvers")
    rel.add_argument("--reason")
    rel.add_argument("--report")

    lt = sub.add_parser("latest", help="print the newest version of a model (in a stage)")
    lt.add_argument("--registry", default=S)
    lt.add_argument("--name", required=True)
    lt.add_argument("--stage")

    ds = sub.add_parser("deploy-sagemaker", help="push an image to ECR and blue/green-deploy a SageMaker endpoint")
    ds.add_argument("--image", required=True, help="local image (pushed to --repository) or an ECR URI")
    ds.add_argument("--repository", help="ECR repository to push a local image to")
    ds.add_argument("--endpoint", required=True)
    ds.add_argument("--role-arn", required=True)
    ds.add_argument("--instance-type", default="ml.m5.large")
    ds.add_argument("--instance-count", type=int, default=1)
    ds.add_argument("--serverless-memory-mb", type=int)
    ds.add_argument("--rollback-alarms", default="", help="comma-separated CloudWatch alarm names")

    ci = sub.add_parser("cicd", help="generate a model CI/CD pipeline")
    cisub = ci.add_subparsers(dest="cicd_cmd", required=True)
    ini = cisub.add_parser("init", help="write a GitHub Actions / GitLab CI pipeline, gates.yml, environments.yml")
    ini.add_argument("--provider", choices=["github", "gitlab"], default="github")
    ini.add_argument("--name", required=True)
    ini.add_argument("--data", required=True)
    ini.add_argument("--target", required=True)
    ini.add_argument("--out", default=".")
    ini.add_argument("--metric", default="roc_auc")
    ini.add_argument("--floor", type=float, default=0.7)
    ini.add_argument("--time-budget", type=int, default=300)
    ini.add_argument("--overwrite", action="store_true")


def _run_pipeline(args) -> int:
    from .. import cicd
    if args.cmd == "train":
        res = cicd.train(args.data, args.target, args.out, test=args.test, test_size=args.test_size, group=args.group,
                         time=args.time, time_budget=args.time_budget,
                         include=args.include.split(",") if args.include else None, report=args.report,
                         max_suspicious_share=args.max_suspicious_share)
        print(json.dumps({k: res[k] for k in ("model", "algorithm", "metrics", "split", "split_issues",
                                                 "suspicious_share")}, indent=2, default=str))
        return 0
    if args.cmd == "gate":
        res = cicd.gate(args.model, args.data, cicd.load_gates(args.gates), report=args.report)
        print("GATES PASSED" if res["passed"] else "GATES FAILED:\n  - " + "\n  - ".join(res["failures"]))
        return 0 if res["passed"] else 1
    if args.cmd == "register":
        if not getattr(args, "registry", None):
            raise SystemExit("register needs --registry")
        res = cicd.register(args.model, args.registry, args.name, metrics=args.metrics, stage=args.stage,
                            description=args.description, sign_key=args.sign_key)
        cicd._write_json(args.report, res)
        print(json.dumps(res))
        return 0
    if args.cmd in ("promote", "latest"):
        from ..registry import ModelRegistry
        if not getattr(args, "registry", None):
            raise SystemExit(f"{args.cmd} needs --registry")
        reg = ModelRegistry(args.registry)
        if args.cmd == "promote":
            mv = reg.promote(args.name, args.version, args.stage, reason=args.reason)
            print(json.dumps({"name": args.name, "version": mv.version, "stage": mv.stage}))
        else:
            print(reg.get(args.name, stage=args.stage).version if args.stage else reg.get(args.name).version)
        return 0
    if args.cmd == "release":
        from ..environments import Environments, ReleaseError
        envs = Environments.from_yaml(args.config)
        try:
            mv = envs.release(args.name, args.version, frm=args.frm, to=args.to, released_by=args.released_by,
                              approved_by=[a for a in args.approved_by.split(",") if a], reason=args.reason)
        except ReleaseError as exc:
            print(f"RELEASE REFUSED: {exc}", file=sys.stderr)
            return 1
        res = {"name": args.name, "environment": args.to, "version": mv.version, "stage": mv.stage}
        cicd._write_json(args.report, res)
        print(json.dumps(res))
        return 0
    if args.cmd == "deploy-sagemaker":
        from . import aws
        image = aws.push_image(args.image, args.repository) if args.repository else args.image
        res = aws.deploy_sagemaker(image, args.endpoint, role_arn=args.role_arn, instance_type=args.instance_type,
                                   instance_count=args.instance_count,
                                   serverless={"memory_mb": args.serverless_memory_mb} if args.serverless_memory_mb else None,
                                   rollback_alarms=[a for a in args.rollback_alarms.split(",") if a])
        print(json.dumps(res, default=str))
        return 0
    written = cicd.init_cicd(args.provider, name=args.name, data=args.data, target=args.target, out=args.out,
                             metric=args.metric, floor=args.floor, budget=args.time_budget, overwrite=args.overwrite)
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
