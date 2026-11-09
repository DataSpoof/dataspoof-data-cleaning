"""MLPilot Hub: the multi-user platform (server + client).

* ``create_app()``     FastAPI server: workspaces, RBAC (viewer / data_scientist / approver / admin),
                       SSO via OIDC + API tokens, runs, model registry with artifact storage
                       (disk / S3 / GCS / Azure), four-eyes production approvals with policy gates,
                       monitoring results, a hash-chained audit log, and a web UI.
* ``HubClient``        Python client (standard library only).
* CLI: ``mlpilot hub serve | create-admin``.

The server needs ``pip install "mlpilot[server]"``; the client needs nothing extra.
"""

from .client import HubClient, HubError

__all__ = ["HubClient", "HubError", "create_app"]


def __getattr__(name):
    if name == "create_app":
        from .app import create_app
        return create_app
    raise AttributeError(name)
