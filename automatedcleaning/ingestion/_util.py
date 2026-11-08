"""Shared helpers: credential redaction, safe SQL identifiers and parameter styles."""

from __future__ import annotations

import datetime as _dt
import re

_URL_PASSWORD = re.compile(r"(?P<head>[A-Za-z][A-Za-z0-9+.\-]*://[^:/@\s]*:)(?P<secret>[^@\s]+)(?P<tail>@)")
_QUERY_SECRET = re.compile(r"(?i)(?P<key>[?&;](?:password|passwd|pwd|token|access_token|api_key|apikey|key|secret|"
                           r"sig|signature|client_secret|sas)=)(?P<secret>[^&;\s]+)")
_NAMED = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def redact(text) -> str:
    """Hide passwords, tokens and keys in a connection string or URL (for logs, reprs, reports)."""
    s = str(text)
    s = _URL_PASSWORD.sub(lambda m: f"{m['head']}***{m['tail']}", s)
    return _QUERY_SECRET.sub(lambda m: f"{m['key']}***", s)


def quote_ident(name: str, quote: str = '"') -> str:
    """Quote a (possibly dotted) SQL identifier safely: ``schema.table`` -> ``"schema"."table"``.

    Embedded quote characters are doubled, so a name can never break out of its quotes."""
    if not isinstance(name, str) or not name.strip() or "\x00" in name:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    close = {"[": "]"}.get(quote, quote)
    parts = name.split(".")
    if any(not p for p in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return ".".join(f"{quote}{p.replace(close, close * 2)}{close}" for p in parts)


def build_select(table: str, *, columns=None, where: dict | None = None, since=None, order_by=None,
                 limit: int | None = None, quote: str = '"') -> tuple:
    """``SELECT`` for a table with every value as a bound ``:name`` parameter (never inlined).

    ``where``: ``{column: value}`` equality filters; ``since``: ``(column, value, inclusive)``."""
    cols = "*" if not columns else ", ".join(quote_ident(c, quote) for c in columns)
    sql = f"SELECT {cols} FROM {quote_ident(table, quote)}"
    params, conds = {}, []
    for i, (col, value) in enumerate((where or {}).items()):
        if value is None:
            conds.append(f"{quote_ident(col, quote)} IS NULL")
        else:
            params[f"w{i}"] = value
            conds.append(f"{quote_ident(col, quote)} = :w{i}")
    if since is not None:
        col, value, inclusive = since if len(since) == 3 else (*since, False)
        if value is not None:
            params["since"] = value
            conds.append(f"{quote_ident(col, quote)} {'>=' if inclusive else '>'} :since")
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    if order_by:
        order = [order_by] if isinstance(order_by, str) else list(order_by)
        sql += " ORDER BY " + ", ".join(quote_ident(c, quote) for c in order)
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql, params


def translate_params(sql: str, params, style: str):
    """Rewrite ``:name`` placeholders for a DB-API ``paramstyle``.

    Returns ``(sql, params)`` with params as a sequence for positional styles."""
    if not params or isinstance(params, (list, tuple)):
        return sql, params
    if style == "named":
        return sql, params
    if style == "pyformat":
        return _NAMED.sub(lambda m: f"%({m[1]})s", sql), params
    order = []

    def repl(m):
        order.append(m[1])
        return "?" if style == "qmark" else ("%s" if style == "format" else f":{len(order)}")
    out = _NAMED.sub(repl, sql)
    missing = [n for n in order if n not in params]
    if missing:
        raise KeyError(f"Missing SQL parameter(s): {missing}")
    return out, [params[n] for n in order]


def to_json_value(v):
    """Watermark values in a JSON-safe, round-trippable form."""
    if isinstance(v, _dt.datetime):
        return {"type": "datetime", "value": v.isoformat()}
    if isinstance(v, _dt.date):
        return {"type": "date", "value": v.isoformat()}
    if hasattr(v, "item"):                       # numpy scalars
        v = v.item()
    return {"type": type(v).__name__, "value": v}


def from_json_value(d):
    if d is None:
        return None
    t, v = d["type"], d["value"]
    if t == "datetime":
        return _dt.datetime.fromisoformat(v)
    if t == "date":
        return _dt.date.fromisoformat(v)
    if t == "Decimal":
        from decimal import Decimal
        return Decimal(str(v))
    return v
