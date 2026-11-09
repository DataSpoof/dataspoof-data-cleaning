"""Isolated dev -> qa -> prod environments with controlled promotion between them.

Each environment has its **own registry** (a separate folder, bucket mount or account), so a
mistake in dev cannot touch production. Models move forward only through :meth:`release`,
which copies the exact artifact and enforces the target environment's rules:

* only to the **next** environment in ``order`` (dev -> qa -> prod; no skipping);
* the source version must be in one of the source's ``release_stages`` (e.g. it passed QA);
* the artifact's checksum is verified, and its **signature** when the target requires it;
* the target's **metric gates** (``min_metrics`` / ``max_metrics``);
* ``approvals`` distinct approvers, none of them the person releasing (four-eyes);
* every release is recorded on both sides (``releases.jsonl``) with who, why, approvals and
  the source version, so production always knows where a model came from.

>>> envs = Environments.from_yaml("environments.yml")
>>> envs.release("churn", 4, frm="qa", to="prod", released_by="ana", approved_by=["bob", "carol"],
...              reason="Q3 refresh, QA sign-off #812")
>>> envs.status("churn")          # which version is where, with provenance

``environments.yml``::

    order: [dev, qa, prod]
    environments:
      dev:  {registry: /mnt/ml/dev-registry}
      qa:   {registry: /mnt/ml/qa-registry, landing_stage: qa, min_metrics: {test_roc_auc: 0.70}}
      prod: {registry: /mnt/ml/prod-registry, landing_stage: staging, approvals: 2,
             require_signature: true, public_key: "vault://secret/ml/signing#public_key",
             min_metrics: {test_roc_auc: 0.75}, release_stages: [production]}

Values may be secret references (``vault://``, ``aws-sm://``, ``env://``, ``file://``).
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from pathlib import Path

import polars as pl

from .registry._store import FileLock, append_jsonl, now, read_jsonl, user
from .registry.models import ModelRegistry, PromotionError


class ReleaseError(PromotionError):
    """A release between environments was refused; nothing was copied."""


@dataclasses.dataclass
class Environment:
    name: str
    registry: ModelRegistry
    landing_stage: str = "staging"           # stage a released model starts in, in this environment
    release_stages: tuple = ("qa", "staging", "production")   # stages a version must be in to leave
    min_metrics: dict = dataclasses.field(default_factory=dict)
    max_metrics: dict = dataclasses.field(default_factory=dict)
    approvals: int = 0
    require_signature: bool = False
    public_key: str | None = None

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Environment":
        from .secretstore import resolve
        reg = d.get("registry")
        if not reg:
            raise ValueError(f"Environment {name!r} needs a registry path")
        return cls(name=name, registry=reg if isinstance(reg, ModelRegistry) else ModelRegistry(resolve(reg)),
                   landing_stage=d.get("landing_stage", "staging"),
                   release_stages=tuple(d.get("release_stages", ("qa", "staging", "production"))),
                   min_metrics=dict(d.get("min_metrics", {})), max_metrics=dict(d.get("max_metrics", {})),
                   approvals=int(d.get("approvals", 0)), require_signature=bool(d.get("require_signature", False)),
                   public_key=d.get("public_key"))


class Environments:
    def __init__(self, environments: dict, order: list | None = None):
        self.envs = {k: (v if isinstance(v, Environment) else Environment.from_dict(k, v))
                     for k, v in environments.items()}
        self.order = list(order or self.envs)
        unknown = [e for e in self.order if e not in self.envs]
        if unknown:
            raise ValueError(f"order mentions unknown environments {unknown}")
        roots = [str(Path(e.registry.root).resolve()) for e in self.envs.values()]
        if len(set(roots)) != len(roots):
            raise ValueError("Each environment needs its own registry (two share a path): isolation is the point")

    @classmethod
    def from_yaml(cls, path) -> "Environments":
        import yaml
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(d["environments"], d.get("order"))

    def __getitem__(self, name: str) -> Environment:
        return self.envs[name]

    def next_of(self, name: str) -> str | None:
        i = self.order.index(name)
        return self.order[i + 1] if i + 1 < len(self.order) else None

    # ── releasing ───────────────────────────────────────────────────────────
    def check(self, name: str, version: int, *, frm: str, to: str, released_by: str | None = None,
              approved_by=(), allow_skip: bool = False) -> list:
        """Everything that would block :meth:`release` (an empty list means it would pass)."""
        if frm not in self.envs or to not in self.envs:
            return [f"unknown environment {frm if frm not in self.envs else to!r}"]
        src, dst = self.envs[frm], self.envs[to]
        failures = []
        if not allow_skip and self.next_of(frm) != to:
            failures.append(f"{frm} releases to {self.next_of(frm)!r}, not {to!r} (environments can't be skipped)")
        try:
            mv = src.registry.get(name, version=version)
        except KeyError as exc:
            return failures + [str(exc).strip("'\"")]
        if mv.stage not in src.release_stages:
            failures.append(f"{name} v{version} is in '{mv.stage}' in {frm}; it must be in one of "
                            f"{list(src.release_stages)} to leave {frm}")
        metrics = mv.metrics or {}
        for metric, minimum in dst.min_metrics.items():
            v = metrics.get(metric)
            if v is None or v < minimum:
                failures.append(f"{metric} = {v} ({to} needs >= {minimum})")
        for metric, maximum in dst.max_metrics.items():
            v = metrics.get(metric)
            if v is None or v > maximum:
                failures.append(f"{metric} = {v} ({to} needs <= {maximum})")
        approvers = {a for a in approved_by if a}
        if released_by:
            approvers.discard(released_by)
        if len(approvers) < dst.approvals:
            failures.append(f"{to} needs {dst.approvals} approval(s) from people other than the releaser "
                            f"(got {len(approvers)})")
        if dst.approvals and not released_by:
            failures.append("released_by is required when approvals are")
        if dst.require_signature:
            from .secretstore import resolve
            from .security import SignatureError, verify_model
            key = resolve(dst.public_key)
            if not key:
                failures.append(f"{to} requires signed models but has no public_key configured")
            else:
                try:
                    verify_model(mv.model_path, key)
                except (SignatureError, FileNotFoundError, ValueError) as exc:
                    failures.append(f"signature check failed: {exc}")
        return failures

    def release(self, name: str, version: int, *, frm: str, to: str, released_by: str | None = None,
                approved_by=(), reason: str | None = None, allow_skip: bool = False):
        """Copy ``name`` v``version`` from environment ``frm`` into ``to`` (see the module docs).
        Returns the new :class:`~mlpilot.registry.models.ModelVersion` in ``to``."""
        failures = self.check(name, version, frm=frm, to=to, released_by=released_by, approved_by=approved_by,
                              allow_skip=allow_skip)
        if failures:
            self._record(self.envs[to], {"event": "release_refused", "model": name, "from": frm, "to": to,
                                         "source_version": version, "released_by": released_by or user(),
                                         "failures": failures})
            raise ReleaseError(f"Release of {name} v{version} {frm} -> {to} refused: " + "; ".join(failures))
        src, dst = self.envs[frm], self.envs[to]
        mv = src.registry.get(name, version=version)
        from .automl import load_model
        load_model(mv.model_path)                         # verifies the stored checksum before anything moves
        with tempfile.TemporaryDirectory() as tmp:        # copy first, so a half-written target can't happen
            staged = Path(tmp) / "model"
            shutil.copytree(mv.model_path, staged)
            approvers = sorted({a for a in approved_by if a and a != released_by})
            tags = {**(mv.tags or {}), "source_env": frm, "source_version": str(version),
                    "source_sha256": str(mv.model_sha256), "released_by": released_by or user(),
                    "approved_by": ",".join(approvers)}
            new = dst.registry.register(staged, name, stage=dst.landing_stage, metrics=dict(mv.metrics or {}),
                                        tags=tags, description=mv.description)
        if new.model_sha256 != mv.model_sha256:          # pragma: no cover - copy corruption
            raise ReleaseError("The copied model's checksum differs from the source")
        event = {"event": "released", "model": name, "from": frm, "to": to, "source_version": version,
                 "target_version": new.version, "landing_stage": dst.landing_stage, "model_sha256": mv.model_sha256,
                 "released_by": released_by or user(), "approved_by": approvers, "reason": reason}
        self._record(dst, event)
        self._record(src, event)
        return new

    def promote_path(self, name: str, version: int, *, released_by: str, approvals: dict | None = None,
                     reason: str | None = None) -> list:
        """Release a version step by step from its current environment to the last one, stopping at
        the first refusal (raised). ``approvals``: ``{env: [approvers]}``. Returns each new version."""
        out = []
        cur_env = self.order[0]
        cur_version = version
        while True:
            nxt = self.next_of(cur_env)
            if nxt is None:
                break
            new = self.release(name, cur_version, frm=cur_env, to=nxt, released_by=released_by,
                               approved_by=(approvals or {}).get(nxt, ()), reason=reason)
            out.append(new)
            nxt_env = self.envs[nxt]
            if new.stage not in nxt_env.release_stages and self.next_of(nxt):
                break                                     # it must pass this environment's own checks first
            cur_env, cur_version = nxt, new.version
        return out

    # ── reading ─────────────────────────────────────────────────────────────
    def _record(self, env: Environment, event: dict) -> None:
        root = Path(env.registry.root)
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / ".releases.lock"):
            append_jsonl(root / "releases.jsonl", {"time": now(), "environment": env.name, **event})

    def releases(self, env: str | None = None) -> pl.DataFrame:
        rows = []
        for e in ([self.envs[env]] if env else self.envs.values()):
            for r in read_jsonl(Path(e.registry.root) / "releases.jsonl"):
                if r.get("environment") == e.name:
                    rows.append({"time": r["time"], "environment": e.name, "event": r["event"], "model": r["model"],
                                 "from": r["from"], "to": r["to"], "source_version": r.get("source_version"),
                                 "target_version": r.get("target_version"), "released_by": r.get("released_by"),
                                 "approved_by": ",".join(r.get("approved_by") or []),
                                 "detail": "; ".join(r.get("failures") or []) or r.get("reason")})
        schema = {"time": pl.String, "environment": pl.String, "event": pl.String, "model": pl.String, "from": pl.String,
                  "to": pl.String, "source_version": pl.Int64, "target_version": pl.Int64, "released_by": pl.String,
                  "approved_by": pl.String, "detail": pl.String}
        return pl.DataFrame(rows, schema=schema).sort("time") if rows else pl.DataFrame(schema=schema)

    def status(self, name: str) -> pl.DataFrame:
        """Every version of ``name`` in every environment, with its stage and where it came from."""
        rows = []
        for env_name in self.order:
            reg = self.envs[env_name].registry
            try:
                versions = reg._versions(name)
            except KeyError:
                versions = []
            for v in versions:
                rows.append({"environment": env_name, "version": v.version, "stage": v.stage,
                             "source": f"{v.tags.get('source_env')} v{v.tags.get('source_version')}"
                             if v.tags.get("source_env") else None, "model_sha256": (v.model_sha256 or "")[:12],
                             "metrics": ", ".join(f"{k}={x:.4g}" for k, x in (v.metrics or {}).items()
                                                  if isinstance(x, (int, float)))})
        schema = {"environment": pl.String, "version": pl.Int64, "stage": pl.String, "source": pl.String,
                  "model_sha256": pl.String, "metrics": pl.String}
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
