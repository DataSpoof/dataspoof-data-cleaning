"""MLPilot Hub: the multi-user platform server (FastAPI).

Run: ``mlpilot hub serve`` (or ``uvicorn mlpilot.hub.app:app``). Configuration
(environment variables):

* ``AC_HUB_DATABASE_URL``   ``postgresql+psycopg://…`` in production (default: SQLite file ``hub.db``)
* ``AC_HUB_ARTIFACTS``      artifact root: a directory or ``s3://`` / ``gs://`` / ``az://`` URL
* ``AC_OIDC_ISSUER`` / ``AC_OIDC_AUDIENCE`` / ``AC_OIDC_JWKS_URL``   single sign-on (OIDC)
* ``AC_HUB_GROUP_ROLES``    JSON: identity-provider group -> ["workspace", "role"]
* ``AC_HUB_MAX_ARTIFACT_MB`` upload limit (default 500)
* ``AC_MODEL_PUBLIC_KEY``   if set, uploaded models must carry a valid signature
* ``AC_HUB_OPA_URL``        Open Policy Agent for access and promotion decisions (see hub.policy)

Any of these may be a secret reference (``vault://secret/hub#database_url``, ``aws-sm://...``,
``file:///run/secrets/...``) instead of the value; see ``mlpilot.secretstore``.

API: ``/api/...`` (OpenAPI docs at ``/docs``); web UI at ``/``.
"""

from __future__ import annotations

import contextvars
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from .db import (ROLES, STAGES, ApiToken, AuditEntry, Membership, ModelVersion, MonitorResult, PromotionRequest, Run,
                 User, Workspace, init_db, utcnow)
from .security import (AuthError, OIDCVerifier, audit, create_token, require_role, role_in, user_from_api_token,
                       user_from_claims, verify_audit_chain)
from .storage import ArtifactStore, sha256, untar_to

_NAME = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$"


# ── request bodies ──────────────────────────────────────────────────────────

class WorkspaceIn(BaseModel):
    name: str = Field(pattern=_NAME)
    require_approval: bool = True
    policy: dict = {}


class MemberIn(BaseModel):
    user: str                      # email, subject or user id
    role: str = Field(pattern="^(viewer|data_scientist|approver|admin)$")


class TokenIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    days: int | None = Field(default=90, ge=1, le=3650)
    for_user: str | None = None    # platform admins can mint tokens for service accounts


class ServiceAccountIn(BaseModel):
    name: str = Field(pattern=_NAME)


class RunIn(BaseModel):
    experiment: str = Field(pattern=_NAME)
    name: str = Field(min_length=1, max_length=200)
    params: dict = {}
    tags: dict = {}


class RunUpdate(BaseModel):
    params: dict | None = None
    metrics: dict | None = None
    tags: dict | None = None
    datasets: list | None = None
    status: str | None = Field(default=None, pattern="^(running|finished|failed)$")


class TransitionIn(BaseModel):
    stage: str = Field(pattern="^(dev|qa|staging|production|archived)$")
    reason: str | None = Field(default=None, max_length=2000)
    evidence: dict = {}


class DecisionIn(BaseModel):
    note: str | None = Field(default=None, max_length=2000)


class MonitorIn(BaseModel):
    version: int | None = None
    rows: int = 0
    drift_share: float | None = None
    metric: str | None = None
    metric_value: float | None = None
    alerts: list = []
    retrain_recommended: bool = False


# ── helpers ─────────────────────────────────────────────────────────────────

def _jsonable(obj) -> Any:
    if isinstance(obj, list):
        return [_jsonable(o) for o in obj]
    if hasattr(obj, "__table__"):
        out = {}
        for c in obj.__table__.columns:
            v = getattr(obj, c.name)
            out[c.name] = v.isoformat() if hasattr(v, "isoformat") else v
        return {k: v for k, v in out.items() if k not in ("token_hash",)}
    return obj


_REQUEST: contextvars.ContextVar = contextvars.ContextVar("ac_hub_request", default=("", ""))


def create_app(*, database_url: str | None = None, artifact_root: str | None = None, oidc=None,
               group_roles: dict | None = None, max_artifact_mb: int | None = None, opa=None) -> FastAPI:
    """``opa``: an :class:`~mlpilot.hub.policy.OPAPolicy` (default: from ``AC_HUB_OPA_URL``)."""
    from ..secretstore import resolve as resolve_secret   # any setting may be a vault:// / aws-sm:// reference
    database_url = resolve_secret(database_url or os.environ.get("AC_HUB_DATABASE_URL", "sqlite:///./hub.db"))
    artifact_root = resolve_secret(artifact_root or os.environ.get("AC_HUB_ARTIFACTS", "./hub-artifacts"))
    if group_roles is None and os.environ.get("AC_HUB_GROUP_ROLES"):
        group_roles = {k: tuple(v) for k, v in json.loads(resolve_secret(os.environ["AC_HUB_GROUP_ROLES"])).items()}
    max_bytes = int(max_artifact_mb or os.environ.get("AC_HUB_MAX_ARTIFACT_MB", "500")) * 1024 * 1024
    Session = init_db(database_url)
    store = ArtifactStore(artifact_root)
    verifier = oidc if oidc is not None else OIDCVerifier.from_env()
    from .policy import OPAPolicy, PolicyUnavailable
    opa = opa if opa is not None else OPAPolicy.from_env()

    @asynccontextmanager
    async def lifespan(_app):
        yield

    app = FastAPI(title="MLPilot Hub", version="2.0", lifespan=lifespan,
                  description="Experiment tracking, model registry, approvals, monitoring and audit.")
    app.state.Session, app.state.store = Session, store

    @app.exception_handler(AuthError)
    async def _auth_error(_req, exc: AuthError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(PolicyUnavailable)
    async def _policy_down(_req, exc: PolicyUnavailable):
        return JSONResponse({"detail": "The policy service is unavailable; the request was refused."}, status_code=503)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        token = _REQUEST.set((request.method, request.url.path))      # for policy decisions
        try:
            resp = await call_next(request)
        finally:
            _REQUEST.reset(token)
        resp.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                             "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
                             "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' "
                                                        "'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"})
        return resp

    def db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def current_user(request: Request, session=Depends(db)) -> User:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise AuthError(401, "Send 'Authorization: Bearer <token>' (an API token or an OIDC token)")
        token = header[7:].strip()
        if token.startswith("ach_"):
            user = user_from_api_token(session, token)
        elif verifier is not None:
            user = user_from_claims(session, verifier.claims(token), group_roles=group_roles)
        else:
            raise AuthError(401, "Invalid token")
        request.state.actor = user.email or user.subject
        return user

    def ip(request: Request) -> str | None:
        return request.client.host if request.client else None

    def authorize(session, user: User, ws: Workspace, minimum: str) -> str:
        """The Hub's role check, then (if configured) OPA - which can only deny, never grant."""
        role = require_role(session, user, ws, minimum)
        if opa is None:
            return role
        method, path = _REQUEST.get()
        allowed, reasons = opa.authorize({
            "user": {"id": user.id, "email": user.email, "subject": user.subject, "is_admin": user.is_admin,
                     "is_service": user.is_service},
            "workspace": ws.name, "role": role, "required_role": minimum,
            "request": {"method": method, "path": path}})
        if not allowed:
            session.rollback()                 # release this request's locks first (SQLite: one writer)
            with Session() as side:            # refusals stay in the audit trail even though the request fails
                audit(side, user.email or user.subject, "access.denied_by_policy", workspace=ws.name,
                      target=f"{method} {path}", details={"reasons": reasons, "required_role": minimum})
                side.commit()
            raise AuthError(403, "Denied by policy" + (": " + "; ".join(reasons) if reasons else ""))
        return role

    def stage_history(session, ws: Workspace, mv: ModelVersion) -> list:
        """Every stage this version has been in, oldest first (from the audit trail)."""
        target = f"{mv.model_name}:v{mv.version}"
        rows = session.scalars(select(AuditEntry).where(AuditEntry.workspace == ws.name, AuditEntry.target == target,
                                                        AuditEntry.action == "model.stage_changed")
                               .order_by(AuditEntry.id))
        return ["dev"] + [(r.details or {}).get("to") for r in rows]

    def workspace(session, name: str) -> Workspace:
        ws = session.scalar(select(Workspace).where(Workspace.name == name))
        if ws is None:
            raise AuthError(404, "Workspace not found")
        return ws

    def version_row(session, ws: Workspace, name: str, version: int) -> ModelVersion:
        mv = session.scalar(select(ModelVersion).where(ModelVersion.workspace_id == ws.id,
                                                       ModelVersion.model_name == name, ModelVersion.version == version))
        if mv is None:
            raise AuthError(404, f"{name} v{version} not found")
        return mv

    def check_gates(ws: Workspace, mv: ModelVersion, *, session=None, to_stage: str = "production",
                    user: User | None = None) -> list:
        """Workspace policy gates (production only) plus OPA promotion rules (any stage change)."""
        failures = []
        builtin = to_stage == "production"
        for metric, minimum in ((ws.policy or {}).get("min_metrics", {}) if builtin else {}).items():
            value = (mv.metrics or {}).get(metric)
            if value is None or value < minimum:
                failures.append(f"{metric} = {value} (policy needs >= {minimum})")
        for metric, maximum in ((ws.policy or {}).get("max_metrics", {}) if builtin else {}).items():
            value = (mv.metrics or {}).get(metric)
            if value is None or value > maximum:
                failures.append(f"{metric} = {value} (policy needs <= {maximum})")
        if builtin and (ws.policy or {}).get("require_signed") and not mv.signed:
            failures.append("model is not signed (policy require_signed)")
        if opa is not None and session is not None:
            failures += opa.promotion_denials({
                "workspace": {"name": ws.name, "policy": ws.policy or {}, "require_approval": ws.require_approval},
                "model": {"name": mv.model_name, "version": mv.version, "stage": mv.stage, "metrics": mv.metrics or {},
                          "signed": mv.signed, "task": mv.task, "description": mv.description,
                          "stage_history": stage_history(session, ws, mv)},
                "to_stage": to_stage,
                "user": {"id": user.id, "email": user.email, "is_service": user.is_service} if user else None})
        return failures

    def apply_stage(session, mv: ModelVersion, stage: str, actor: str, reason, request: Request):
        ws_name = session.get(Workspace, mv.workspace_id).name
        if stage == "production":
            for old in session.scalars(select(ModelVersion).where(
                    ModelVersion.workspace_id == mv.workspace_id, ModelVersion.model_name == mv.model_name,
                    ModelVersion.stage == "production", ModelVersion.id != mv.id)):
                old.stage = "archived"
                audit(session, actor, "model.archived", workspace=ws_name, target=f"{old.model_name}:v{old.version}",
                      details={"replaced_by": mv.version}, ip=ip(request))
        before, mv.stage = mv.stage, stage
        audit(session, actor, "model.stage_changed", workspace=ws_name, target=f"{mv.model_name}:v{mv.version}",
              details={"from": before, "to": stage, "reason": reason}, ip=ip(request))

    # ── public ──────────────────────────────────────────────────────────────
    @app.get("/health")
    def health(session=Depends(db)):
        session.execute(select(1))
        return {"status": "ok", "database": "ok"}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui_page():
        return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")

    @app.get("/ui.js", include_in_schema=False)
    def ui_js():
        return Response((Path(__file__).parent / "ui.js").read_text(encoding="utf-8"), media_type="text/javascript")

    # ── identity ────────────────────────────────────────────────────────────
    @app.get("/api/me")
    def me(user: User = Depends(current_user), session=Depends(db)):
        rows = session.execute(select(Workspace.name, Membership.role).join(
            Membership, Membership.workspace_id == Workspace.id).where(Membership.user_id == user.id)).all()
        workspaces = {n: r for n, r in rows}
        if user.is_admin:
            workspaces = {w.name: "admin" for w in session.scalars(select(Workspace))}
        return {**_jsonable(user), "workspaces": workspaces}

    @app.post("/api/tokens", status_code=201)
    def new_token(body: TokenIn, request: Request, user: User = Depends(current_user), session=Depends(db)):
        owner = user
        if body.for_user:
            if not user.is_admin:
                raise AuthError(403, "Only platform admins can create tokens for other accounts")
            owner = session.scalar(select(User).where((User.id == body.for_user) | (User.subject == body.for_user)
                                                      | (User.email == body.for_user)))
            if owner is None:
                raise AuthError(404, "User not found")
        token = create_token(session, owner, body.name, body.days)
        audit(session, request.state.actor, "token.created", target=owner.subject, details={"name": body.name},
              ip=ip(request))
        return {"token": token, "note": "Store it now: it is not shown again."}

    @app.get("/api/tokens")
    def list_tokens(user: User = Depends(current_user), session=Depends(db)):
        return _jsonable(list(session.scalars(select(ApiToken).where(ApiToken.user_id == user.id))))

    @app.delete("/api/tokens/{token_id}")
    def revoke_token(token_id: str, request: Request, user: User = Depends(current_user), session=Depends(db)):
        tok = session.get(ApiToken, token_id)
        if tok is None or (tok.user_id != user.id and not user.is_admin):
            raise AuthError(404, "Token not found")
        tok.revoked = True
        audit(session, request.state.actor, "token.revoked", target=tok.prefix, ip=ip(request))
        return {"revoked": token_id}

    @app.post("/api/service-accounts", status_code=201)
    def service_account(body: ServiceAccountIn, request: Request, user: User = Depends(current_user),
                        session=Depends(db)):
        if not user.is_admin:
            raise AuthError(403, "Only platform admins can create service accounts")
        sa = User(subject=f"service:{body.name}", name=body.name, is_service=True)
        session.add(sa)
        session.flush()
        token = create_token(session, sa, f"{body.name}-default", 365)
        audit(session, request.state.actor, "service_account.created", target=sa.subject, ip=ip(request))
        return {"id": sa.id, "subject": sa.subject, "token": token}

    # ── workspaces & members ────────────────────────────────────────────────
    @app.post("/api/workspaces", status_code=201)
    def create_workspace(body: WorkspaceIn, request: Request, user: User = Depends(current_user), session=Depends(db)):
        if not user.is_admin:
            raise AuthError(403, "Only platform admins can create workspaces")
        if session.scalar(select(Workspace).where(Workspace.name == body.name)):
            raise HTTPException(409, "Workspace exists")
        ws = Workspace(name=body.name, require_approval=body.require_approval, policy=body.policy)
        session.add(ws)
        session.flush()
        audit(session, request.state.actor, "workspace.created", workspace=ws.name,
              details={"require_approval": ws.require_approval, "policy": ws.policy}, ip=ip(request))
        return _jsonable(ws)

    @app.get("/api/workspaces")
    def list_workspaces(user: User = Depends(current_user), session=Depends(db)):
        rows = session.scalars(select(Workspace).order_by(Workspace.name))
        return [{**_jsonable(w), "role": role_in(session, user, w)} for w in rows if role_in(session, user, w)]

    @app.put("/api/workspaces/{ws_name}/members")
    def set_member(ws_name: str, body: MemberIn, request: Request, user: User = Depends(current_user),
                   session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "admin")
        target = session.scalar(select(User).where((User.id == body.user) | (User.subject == body.user)
                                                   | (User.email == body.user)))
        if target is None:                      # pre-provision by email; matched on first SSO login
            target = User(subject=f"pending:{body.user}", email=body.user, name=body.user)
            session.add(target)
            session.flush()
        m = session.scalar(select(Membership).where(Membership.user_id == target.id, Membership.workspace_id == ws.id))
        before = m.role if m else None
        if m is None:
            session.add(Membership(user_id=target.id, workspace_id=ws.id, role=body.role))
        else:
            m.role = body.role
        audit(session, request.state.actor, "member.role_set", workspace=ws.name, target=target.email or target.subject,
              details={"from": before, "to": body.role}, ip=ip(request))
        return {"user": target.id, "role": body.role}

    @app.get("/api/workspaces/{ws_name}/members")
    def list_members(ws_name: str, user: User = Depends(current_user), session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        rows = session.scalars(select(Membership).where(Membership.workspace_id == ws.id))
        return [{"user": m.user.email or m.user.subject, "user_id": m.user_id, "role": m.role} for m in rows]

    @app.delete("/api/workspaces/{ws_name}/members/{user_id}")
    def remove_member(ws_name: str, user_id: str, request: Request, user: User = Depends(current_user),
                      session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "admin")
        m = session.scalar(select(Membership).where(Membership.user_id == user_id, Membership.workspace_id == ws.id))
        if m is None:
            raise AuthError(404, "Member not found")
        session.delete(m)
        audit(session, request.state.actor, "member.removed", workspace=ws.name, target=user_id, ip=ip(request))
        return {"removed": user_id}

    # ── runs ────────────────────────────────────────────────────────────────
    @app.post("/api/workspaces/{ws_name}/runs", status_code=201)
    def create_run(ws_name: str, body: RunIn, user: User = Depends(current_user), session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "data_scientist")
        run = Run(workspace_id=ws.id, experiment=body.experiment, name=body.name, params=body.params, tags=body.tags,
                  metrics={}, datasets=[], created_by=user.id)
        session.add(run)
        session.flush()
        return _jsonable(run)

    def run_row(session, user, run_id: str, minimum: str) -> Run:
        run = session.get(Run, run_id)
        if run is None:
            raise AuthError(404, "Run not found")
        authorize(session, user, session.get(Workspace, run.workspace_id), minimum)
        return run

    @app.patch("/api/runs/{run_id}")
    def update_run(run_id: str, body: RunUpdate, user: User = Depends(current_user), session=Depends(db)):
        run = run_row(session, user, run_id, "data_scientist")
        if run.status != "running" and body.status is None:
            raise HTTPException(409, f"Run has ended ({run.status})")
        if body.params:
            run.params = {**run.params, **body.params}
        if body.metrics:
            run.metrics = {**run.metrics, **body.metrics}
        if body.tags:
            run.tags = {**run.tags, **body.tags}
        if body.datasets:
            run.datasets = [*run.datasets, *body.datasets]
        if body.status and body.status != "running":
            run.status, run.ended_at = body.status, utcnow()
        return _jsonable(run)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, user: User = Depends(current_user), session=Depends(db)):
        return _jsonable(run_row(session, user, run_id, "viewer"))

    @app.get("/api/workspaces/{ws_name}/runs")
    def list_runs(ws_name: str, experiment: str | None = None, limit: int = 200, user: User = Depends(current_user),
                  session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        q = select(Run).where(Run.workspace_id == ws.id)
        if experiment:
            q = q.where(Run.experiment == experiment)
        return _jsonable(list(session.scalars(q.order_by(Run.started_at.desc()).limit(min(limit, 1000)))))

    # ── models ──────────────────────────────────────────────────────────────
    @app.post("/api/workspaces/{ws_name}/models/{model_name}/versions", status_code=201)
    def upload_version(ws_name: str, model_name: str, request: Request, artifact: UploadFile = File(...),
                       description: str | None = Form(None), run_id: str | None = Form(None),
                       metrics: str | None = Form(None), user: User = Depends(current_user), session=Depends(db)):
        import re
        if not re.match(_NAME, model_name):
            raise HTTPException(422, "Invalid model name")
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "data_scientist")
        blob = artifact.file.read(max_bytes + 1)
        if len(blob) > max_bytes:
            raise HTTPException(413, f"Artifact larger than {max_bytes // (1024 * 1024)} MB")
        with tempfile.TemporaryDirectory() as tmp:            # inspect it; never unpickle on the server
            try:
                root = untar_to(blob, tmp)
            except Exception as exc:
                raise HTTPException(422, f"Not a valid model artifact (.tar.gz): {exc}") from None
            mf = root / "manifest.json"
            if not mf.exists():
                raise HTTPException(422, "Artifact has no manifest.json (save the model with model.save())")
            manifest = json.loads(mf.read_text(encoding="utf-8"))
            model_file = root / str(manifest.get("model_file", ""))
            if not model_file.is_file() or sha256(model_file.read_bytes()) != manifest.get("model_sha256"):
                raise HTTPException(422, "Model file checksum doesn't match its manifest")
            signed = (root / "signature.json").exists()
            if os.environ.get("AC_MODEL_PUBLIC_KEY") or (ws.policy or {}).get("require_signed"):
                if not os.environ.get("AC_MODEL_PUBLIC_KEY"):
                    raise HTTPException(500, "require_signed policy set but AC_MODEL_PUBLIC_KEY is not configured")
                from ..security import SignatureError, verify_model
                try:
                    verify_model(root, os.environ["AC_MODEL_PUBLIC_KEY"])
                except SignatureError as exc:
                    raise HTTPException(422, f"Signature check failed: {exc}") from None
        if run_id and session.get(Run, run_id) is None:
            raise HTTPException(422, "Unknown run_id")
        latest = session.scalar(select(func.max(ModelVersion.version)).where(
            ModelVersion.workspace_id == ws.id, ModelVersion.model_name == model_name)) or 0
        version = latest + 1
        uri = store.put(blob, ws.id, model_name, f"v{version}.tar.gz")
        task = manifest.get("task", {}).get("task") if isinstance(manifest.get("task"), dict) else manifest.get("kind")
        mv = ModelVersion(workspace_id=ws.id, model_name=model_name, version=version, stage="dev", task=task,
                          target=manifest.get("target"), metrics=json.loads(metrics) if metrics else {},
                          description=description, run_id=run_id, artifact_uri=uri, artifact_sha256=sha256(blob),
                          model_sha256=manifest.get("model_sha256"), signed=signed, created_by=user.id)
        session.add(mv)
        session.flush()
        audit(session, request.state.actor, "model.registered", workspace=ws.name, target=f"{model_name}:v{version}",
              details={"sha256": mv.artifact_sha256, "signed": signed, "run_id": run_id}, ip=ip(request))
        return _jsonable(mv)

    @app.get("/api/workspaces/{ws_name}/models")
    def list_models(ws_name: str, user: User = Depends(current_user), session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        out: dict = {}
        for mv in session.scalars(select(ModelVersion).where(ModelVersion.workspace_id == ws.id)
                                  .order_by(ModelVersion.version)):
            m = out.setdefault(mv.model_name, {"name": mv.model_name, "versions": 0, "latest": None, "production": None,
                                               "staging": None, "task": mv.task})
            m["versions"] += 1
            m["latest"] = mv.version
            if mv.stage in ("production", "staging"):
                m[mv.stage] = mv.version
        return list(out.values())

    @app.get("/api/workspaces/{ws_name}/models/{model_name}/versions")
    def list_versions(ws_name: str, model_name: str, user: User = Depends(current_user), session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        return _jsonable(list(session.scalars(select(ModelVersion).where(
            ModelVersion.workspace_id == ws.id, ModelVersion.model_name == model_name).order_by(ModelVersion.version))))

    @app.get("/api/workspaces/{ws_name}/models/{model_name}/resolve")
    def resolve(ws_name: str, model_name: str, stage: str = "production", user: User = Depends(current_user),
                session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        mv = session.scalar(select(ModelVersion).where(ModelVersion.workspace_id == ws.id,
                                                       ModelVersion.model_name == model_name,
                                                       ModelVersion.stage == stage)
                            .order_by(ModelVersion.version.desc()).limit(1))
        if mv is None:
            raise AuthError(404, f"No {model_name} version in {stage}")
        return _jsonable(mv)

    @app.get("/api/workspaces/{ws_name}/models/{model_name}/versions/{version}/artifact")
    def download(ws_name: str, model_name: str, version: int, request: Request, user: User = Depends(current_user),
                 session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        mv = version_row(session, ws, model_name, version)
        blob = store.get(mv.artifact_uri)
        if sha256(blob) != mv.artifact_sha256:
            raise HTTPException(500, "Stored artifact failed its integrity check")
        audit(session, request.state.actor, "model.downloaded", workspace=ws.name, target=f"{model_name}:v{version}",
              ip=ip(request))
        return Response(blob, media_type="application/gzip",
                        headers={"Content-Disposition": f'attachment; filename="{model_name}-v{version}.tar.gz"',
                                 "X-Artifact-SHA256": mv.artifact_sha256})

    # ── promotion with approvals ────────────────────────────────────────────
    @app.post("/api/workspaces/{ws_name}/models/{model_name}/versions/{version}/transitions")
    def transition(ws_name: str, model_name: str, version: int, body: TransitionIn, request: Request,
                   user: User = Depends(current_user), session=Depends(db)):
        ws = workspace(session, ws_name)
        mv = version_row(session, ws, model_name, version)
        failures = check_gates(ws, mv, session=session, to_stage=body.stage, user=user) \
            if body.stage == "production" or opa is not None else []
        if failures:
            # the request's own transaction is rolled back by the error below, so record the blocked
            # attempt in a separate, committed transaction: refusals must stay in the audit trail
            session.rollback()                    # release this request's locks first (SQLite: one writer)
            with Session() as side:
                audit(side, request.state.actor, "model.promotion_blocked", workspace=ws.name,
                      target=f"{model_name}:v{version}", details={"failures": failures, "to": body.stage},
                      ip=ip(request))
                side.commit()
            raise HTTPException(422, {"message": "Blocked by workspace policy", "failures": failures})
        if body.stage == "production":
            if ws.require_approval:
                authorize(session, user, ws, "data_scientist")
                pending = session.scalar(select(PromotionRequest).where(PromotionRequest.model_version_id == mv.id,
                                                                         PromotionRequest.status == "pending"))
                if pending:
                    raise HTTPException(409, f"A promotion request is already pending ({pending.id})")
                req = PromotionRequest(model_version_id=mv.id, to_stage="production", reason=body.reason,
                                       evidence={**body.evidence, "metrics": mv.metrics}, requested_by=user.id)
                session.add(req)
                session.flush()
                audit(session, request.state.actor, "promotion.requested", workspace=ws.name,
                      target=f"{model_name}:v{version}", details={"request": req.id, "reason": body.reason},
                      ip=ip(request))
                return JSONResponse({"status": "pending_approval", "request": _jsonable(req)}, status_code=202)
            authorize(session, user, ws, "approver")
        else:
            authorize(session, user, ws, "data_scientist")
        apply_stage(session, mv, body.stage, request.state.actor, body.reason, request)
        return {"status": "applied", "version": _jsonable(mv)}

    @app.get("/api/workspaces/{ws_name}/promotions")
    def list_promotions(ws_name: str, status: str | None = "pending", user: User = Depends(current_user),
                        session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        q = select(PromotionRequest).join(ModelVersion).where(ModelVersion.workspace_id == ws.id)
        if status:
            q = q.where(PromotionRequest.status == status)
        return [{**_jsonable(r), "model": r.model_version.model_name, "version": r.model_version.version}
                for r in session.scalars(q.order_by(PromotionRequest.created_at.desc()))]

    def decide(request_id: str, approve: bool, body: DecisionIn, request: Request, user: User, session):
        req = session.get(PromotionRequest, request_id)
        if req is None:
            raise AuthError(404, "Request not found")
        mv = req.model_version
        ws = session.get(Workspace, mv.workspace_id)
        authorize(session, user, ws, "approver")
        if req.status != "pending":
            raise HTTPException(409, f"Request already {req.status}")
        if approve and req.requested_by == user.id:
            raise AuthError(403, "Four-eyes rule: someone other than the requester must approve")
        if approve:
            failures = check_gates(ws, mv, session=session, to_stage=req.to_stage, user=user)   # policy may have changed
            if failures:
                raise HTTPException(422, {"message": "Blocked by workspace policy", "failures": failures})
        req.status, req.decided_by, req.decided_at, req.decision_note = (
            "approved" if approve else "rejected", user.id, utcnow(), body.note)
        audit(session, request.state.actor, f"promotion.{req.status}", workspace=ws.name,
              target=f"{mv.model_name}:v{mv.version}", details={"request": req.id, "note": body.note}, ip=ip(request))
        if approve:
            apply_stage(session, mv, req.to_stage, request.state.actor, f"approved request {req.id}", request)
        return {"status": req.status, "version": _jsonable(mv)}

    @app.post("/api/promotions/{request_id}/approve")
    def approve(request_id: str, body: DecisionIn, request: Request, user: User = Depends(current_user),
                session=Depends(db)):
        return decide(request_id, True, body, request, user, session)

    @app.post("/api/promotions/{request_id}/reject")
    def reject(request_id: str, body: DecisionIn, request: Request, user: User = Depends(current_user),
               session=Depends(db)):
        return decide(request_id, False, body, request, user, session)

    # ── monitoring ──────────────────────────────────────────────────────────
    @app.post("/api/workspaces/{ws_name}/models/{model_name}/monitoring", status_code=201)
    def post_monitoring(ws_name: str, model_name: str, body: MonitorIn, request: Request,
                        user: User = Depends(current_user), session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "data_scientist")
        row = MonitorResult(workspace_id=ws.id, model_name=model_name, **body.model_dump())
        session.add(row)
        if body.retrain_recommended:
            audit(session, request.state.actor, "monitoring.retrain_recommended", workspace=ws.name,
                  target=model_name, details={"alerts": body.alerts[:20]}, ip=ip(request))
        session.flush()
        return _jsonable(row)

    @app.get("/api/workspaces/{ws_name}/models/{model_name}/monitoring")
    def get_monitoring(ws_name: str, model_name: str, limit: int = 200, user: User = Depends(current_user),
                       session=Depends(db)):
        ws = workspace(session, ws_name)
        authorize(session, user, ws, "viewer")
        return _jsonable(list(session.scalars(select(MonitorResult).where(
            MonitorResult.workspace_id == ws.id, MonitorResult.model_name == model_name)
            .order_by(MonitorResult.checked_at.desc()).limit(min(limit, 2000)))))

    # ── audit ───────────────────────────────────────────────────────────────
    def audit_query(session, user: User, ws_name: str | None):
        if ws_name:
            ws = workspace(session, ws_name)
            authorize(session, user, ws, "admin")
            return select(AuditEntry).where(AuditEntry.workspace == ws.name)
        if not user.is_admin:
            raise AuthError(403, "Only platform admins can read the full audit log")
        return select(AuditEntry)

    @app.get("/api/audit")
    def get_audit(workspace_name: str | None = None, limit: int = 500, user: User = Depends(current_user),
                  session=Depends(db)):
        q = audit_query(session, user, workspace_name)
        return _jsonable(list(session.scalars(q.order_by(AuditEntry.id.desc()).limit(min(limit, 5000)))))

    @app.get("/api/audit/verify")
    def verify_audit(user: User = Depends(current_user), session=Depends(db)):
        if not user.is_admin:
            raise AuthError(403, "Only platform admins can verify the audit log")
        return verify_audit_chain(session)

    @app.get("/api/audit/export", response_class=PlainTextResponse)
    def export_audit(workspace_name: str | None = None, user: User = Depends(current_user), session=Depends(db)):
        q = audit_query(session, user, workspace_name)
        lines = [json.dumps(_jsonable(e), default=str) for e in session.scalars(q.order_by(AuditEntry.id))]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="application/x-ndjson")

    return app


def bootstrap_admin(database_url: str | None = None, name: str = "admin", email: str | None = None) -> str:
    """Create a platform admin (if missing) and return a new API token for it (printed once)."""
    Session = init_db(database_url or os.environ.get("AC_HUB_DATABASE_URL", "sqlite:///./hub.db"))
    with Session() as session:
        subject = f"local:{name}"
        user = session.scalar(select(User).where(User.subject == subject))
        if user is None:
            user = User(subject=subject, name=name, email=email, is_admin=True)
            session.add(user)
            session.flush()
        token = create_token(session, user, "bootstrap", 30)
        audit(session, "cli", "admin.bootstrap", target=subject)
        session.commit()
    return token


def __getattr__(name):            # ``uvicorn mlpilot.hub.app:app`` builds from the environment
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
