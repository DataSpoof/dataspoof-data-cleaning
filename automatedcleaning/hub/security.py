"""Hub authentication (OIDC bearer tokens + API tokens), RBAC and the hash-chained audit log."""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os
import secrets

from sqlalchemy import select

from .db import ROLES, ApiToken, AuditEntry, Membership, User, Workspace, utcnow

TOKEN_PREFIX = "ach_"


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ── API tokens ──────────────────────────────────────────────────────────────

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_token(session, user: User, name: str, days: int | None = 90) -> str:
    """Create an API token; returns the token (shown once — only its hash is stored)."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    expires = utcnow() + _dt.timedelta(days=days) if days else None
    session.add(ApiToken(user_id=user.id, name=name, prefix=token[:12], token_hash=hash_token(token),
                         expires_at=expires))
    return token


def user_from_api_token(session, token: str) -> User:
    row = session.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(token)))
    if row is None or row.revoked:
        raise AuthError(401, "Invalid or revoked token")
    expires = row.expires_at
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=_dt.timezone.utc)
    if expires is not None and expires < utcnow():
        raise AuthError(401, "Token expired")
    row.last_used_at = utcnow()
    return row.user


# ── OIDC (Okta, Azure AD / Entra ID, Google, Keycloak, Auth0 …) ─────────────

class OIDCVerifier:
    """Validates OIDC access/ID tokens (JWT) against the issuer's published keys (JWKS).

    Configure with ``AC_OIDC_ISSUER`` and ``AC_OIDC_AUDIENCE`` (and optionally ``AC_OIDC_JWKS_URL``;
    by default it is discovered from ``<issuer>/.well-known/openid-configuration``)."""

    def __init__(self, issuer: str, audience: str, jwks_url: str | None = None, *, leeway: int = 30):
        import jwt
        self.issuer, self.audience, self.leeway = issuer.rstrip("/"), audience, leeway
        if jwks_url is None:
            import urllib.request
            with urllib.request.urlopen(f"{self.issuer}/.well-known/openid-configuration", timeout=10) as r:  # noqa: S310
                jwks_url = json.loads(r.read())["jwks_uri"]
        self._jwks = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)

    def claims(self, token: str) -> dict:
        import jwt
        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            return jwt.decode(token, key, algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "PS256"],
                              audience=self.audience, issuer=self.issuer, leeway=self.leeway,
                              options={"require": ["exp", "iat", "sub"]})
        except jwt.PyJWTError as exc:
            raise AuthError(401, f"Invalid identity token: {exc}") from None

    @classmethod
    def from_env(cls):
        from ..secretstore import resolve
        issuer = resolve(os.environ.get("AC_OIDC_ISSUER"))
        if not issuer:
            return None
        return cls(issuer, resolve(os.environ.get("AC_OIDC_AUDIENCE", "automatedcleaning-hub")),
                   resolve(os.environ.get("AC_OIDC_JWKS_URL")))


def user_from_claims(session, claims: dict, *, group_roles: dict | None = None, group_claim: str = "groups") -> User:
    """Find or create (just-in-time provisioning) the user for verified OIDC claims.

    ``group_roles`` maps identity-provider groups to workspace roles, e.g.
    ``{"ml-admins": ("*", "admin"), "churn-team": ("churn", "data_scientist")}`` — memberships
    are synced on every login, so access follows the identity provider."""
    subject = f"oidc:{claims['sub']}"
    user = session.scalar(select(User).where(User.subject == subject))
    if user is None:
        user = User(subject=subject, email=claims.get("email"), name=claims.get("name") or claims.get("email"))
        session.add(user)
        session.flush()
    else:
        user.email = claims.get("email") or user.email
        user.name = claims.get("name") or user.name
    for group in claims.get(group_claim, []) or []:
        mapping = (group_roles or {}).get(group)
        if not mapping:
            continue
        ws_name, role = mapping
        if ws_name == "*" and role == "admin":
            user.is_admin = True
            continue
        ws = session.scalar(select(Workspace).where(Workspace.name == ws_name))
        if ws is None:
            continue
        m = session.scalar(select(Membership).where(Membership.user_id == user.id, Membership.workspace_id == ws.id))
        if m is None:
            session.add(Membership(user_id=user.id, workspace_id=ws.id, role=role))
        elif ROLES[role] > ROLES[m.role]:
            m.role = role
    return user


# ── RBAC ────────────────────────────────────────────────────────────────────

def role_in(session, user: User, workspace: Workspace) -> str | None:
    if user.is_admin:
        return "admin"
    m = session.scalar(select(Membership).where(Membership.user_id == user.id,
                                                Membership.workspace_id == workspace.id))
    return m.role if m else None


def require_role(session, user: User, workspace: Workspace, minimum: str) -> str:
    role = role_in(session, user, workspace)
    if role is None:
        raise AuthError(404, "Workspace not found")              # don't reveal workspaces you can't see
    if ROLES[role] < ROLES[minimum]:
        raise AuthError(403, f"This needs the '{minimum}' role in workspace '{workspace.name}' (you are '{role}')")
    return role


# ── audit log (tamper-evident: every entry hashes the previous one) ─────────

GENESIS = "0" * 64


def _entry_hash(prev: str, ts: str, actor: str, action: str, workspace, target, details: dict) -> str:
    body = json.dumps({"prev": prev, "ts": ts, "actor": actor, "action": action, "workspace": workspace,
                       "target": target, "details": details}, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def audit(session, actor: str, action: str, *, workspace: str | None = None, target: str | None = None,
          details: dict | None = None, ip: str | None = None) -> AuditEntry:
    if session.get_bind().dialect.name == "postgresql":
        # serialise appends across server processes so the chain can't fork (released at commit)
        from sqlalchemy import text
        session.execute(text("SELECT pg_advisory_xact_lock(7426001)"))
    last = session.scalar(select(AuditEntry).order_by(AuditEntry.id.desc()).limit(1))
    prev = last.hash if last else GENESIS
    ts = utcnow().replace(microsecond=0)
    details = json.loads(json.dumps(details or {}, default=str))
    entry = AuditEntry(ts=ts, actor=actor, action=action, workspace=workspace, target=target, details=details, ip=ip,
                       prev_hash=prev, hash=_entry_hash(prev, ts.isoformat(), actor, action, workspace, target, details))
    session.add(entry)
    session.flush()
    return entry


def verify_audit_chain(session) -> dict:
    """Recompute every hash. Any edited, deleted or reordered entry breaks the chain."""
    prev, count = GENESIS, 0
    for e in session.scalars(select(AuditEntry).order_by(AuditEntry.id)):
        ts = e.ts if e.ts.tzinfo else e.ts.replace(tzinfo=_dt.timezone.utc)
        expected = _entry_hash(prev, ts.isoformat(), e.actor, e.action, e.workspace, e.target, e.details)
        if e.prev_hash != prev or not hmac.compare_digest(e.hash, expected):
            return {"valid": False, "entries": count, "broken_at": e.id}
        prev, count = e.hash, count + 1
    return {"valid": True, "entries": count, "head": prev}
