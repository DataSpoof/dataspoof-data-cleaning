"""Policy as code for the Hub with Open Policy Agent (OPA).

The Hub's own roles (viewer < data_scientist < approver < admin) are always enforced. With OPA
configured, every authorised request is *also* sent to OPA, which can deny it (deny-overrides),
and every promotion is checked against OPA's promotion rules in addition to the workspace
policy. Policies live in Rego files you version and review like code; ``deploy/opa`` has a
ready-to-use set with tests (``opa test deploy/opa/policies``).

Configuration: ``AC_HUB_OPA_URL`` (e.g. ``http://opa:8181``), optional ``AC_HUB_OPA_TOKEN``
(bearer token if OPA runs with authentication), ``AC_HUB_OPA_PACKAGE`` (default
``automatedcleaning``) and ``AC_HUB_OPA_FAIL_OPEN=1`` to allow requests when OPA is unreachable
(default: fail closed - the request is refused with 503).

Queried documents:

* ``data.<package>.authz.allow`` (bool) and ``data.<package>.authz.reasons`` (list of strings).
  Input: ``user`` (id, email, is_admin, is_service), ``workspace``, ``role``,
  ``required_role``, ``request`` (method, path).
* ``data.<package>.promotion.deny`` (set of messages). Input: ``workspace`` (name, policy,
  require_approval), ``model`` (name, version, stage, metrics, signed, task, description,
  stage_history), ``to_stage``, ``user``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class PolicyUnavailable(RuntimeError):
    """OPA could not be reached (or answered with an error) and the policy fails closed."""


class OPAPolicy:
    def __init__(self, url: str, *, package: str = "automatedcleaning", token: str | None = None,
                 timeout: float = 2.0, fail_open: bool = False):
        self.url, self.package = url.rstrip("/"), package.replace(".", "/")
        self.token, self.timeout, self.fail_open = token, timeout, fail_open

    @classmethod
    def from_env(cls) -> "OPAPolicy | None":
        from ..secretstore import resolve
        url = resolve(os.environ.get("AC_HUB_OPA_URL"))
        if not url:
            return None
        return cls(url, package=os.environ.get("AC_HUB_OPA_PACKAGE", "automatedcleaning"),
                   token=resolve(os.environ.get("AC_HUB_OPA_TOKEN")),
                   fail_open=os.environ.get("AC_HUB_OPA_FAIL_OPEN", "0") == "1")

    def query(self, document: str, input_: dict):
        """The value of ``data.<package>.<document>`` for ``input_`` (``None`` if undefined)."""
        req = urllib.request.Request(f"{self.url}/v1/data/{self.package}/{document.replace('.', '/')}",
                                     data=json.dumps({"input": input_}, default=str).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read()).get("result")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise PolicyUnavailable(f"OPA at {self.url} did not answer: {getattr(exc, 'reason', exc)}") from None

    def authorize(self, input_: dict) -> tuple:
        """``(allowed, reasons)``. When OPA is unreachable: ``fail_open`` decides."""
        try:
            allowed = self.query("authz.allow", input_)
            reasons = (self.query("authz.reasons", input_) or []) if not allowed else []
        except PolicyUnavailable:
            if self.fail_open:
                return True, []
            raise
        return bool(allowed), sorted(reasons) if isinstance(reasons, list) else []

    def promotion_denials(self, input_: dict) -> list:
        try:
            deny = self.query("promotion.deny", input_)
        except PolicyUnavailable:
            if self.fail_open:
                return []
            raise
        return sorted(deny or [])
