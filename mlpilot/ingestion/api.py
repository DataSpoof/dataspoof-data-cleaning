"""REST/JSON APIs: authentication, pagination, retries with back-off, rate limiting.

Pagination styles: ``"page"`` (?page=1,2,…), ``"offset"`` (?offset=0&limit=100),
``"cursor"`` (a cursor value from the body), ``"next"`` (a next-page URL in the body) and
``"link"`` (RFC 8288 ``Link: <…>; rel="next"`` header). Next-page URLs must stay on the same
origin as ``base_url`` so a token is never sent to another host.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import polars as pl

from ._util import redact

MAX_RESPONSE_BYTES = 200 * 1024 * 1024
_LINK_NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?', re.I)


def _dig(obj, path: str | None):
    if not path:
        return obj
    for part in path.split("."):
        if obj is None:
            return None
        obj = obj[int(part)] if isinstance(obj, list) and part.isdigit() else (obj.get(part) if isinstance(obj, dict)
                                                                              else None)
    return obj


class APIError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} from {redact(url)}: {body[:300]}")
        self.status = status


class RESTSource:
    """A JSON API.

    >>> api = RESTSource("https://api.example.com/v1", token_env="EXAMPLE_TOKEN")
    >>> api.fetch("orders", records="data", pagination="cursor", cursor_param="after",
    ...           cursor_path="meta.next_cursor", params={"status": "paid"})
    """

    def __init__(self, base_url: str, *, headers: dict | None = None, token: str | None = None,
                 token_env: str | None = None, auth_header: str = "Authorization", auth_scheme: str | None = "Bearer",
                 params: dict | None = None, timeout: float = 30.0, retries: int = 4, backoff: float = 1.0,
                 rate_limit: float | None = None, verify_origin: bool = True):
        if not base_url.lower().startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        self.base_url = base_url.rstrip("/")
        self._origin = urllib.parse.urlsplit(self.base_url)[:2]
        self.headers = {"Accept": "application/json", "User-Agent": "mlpilot-ingest", **(headers or {})}
        token = token or (os.environ.get(token_env) if token_env else None)
        if token_env and not token:
            raise ValueError(f"Environment variable {token_env} is not set")
        if token:
            self.headers[auth_header] = f"{auth_scheme} {token}" if auth_scheme else token
        self.params, self.timeout, self.retries, self.backoff = dict(params or {}), timeout, retries, backoff
        self.min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self.verify_origin = verify_origin
        self._last = 0.0
        self.requests_made = 0

    def _url(self, endpoint: str, params: dict | None) -> str:
        absolute = endpoint.lower().startswith(("http://", "https://"))
        url = endpoint if absolute else (f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url)
        if self.verify_origin and urllib.parse.urlsplit(url)[:2] != self._origin:
            raise ValueError(f"Refusing to call {redact(url)}: it is not on {self._origin[0]}://{self._origin[1]}")
        # next-page URLs from the server already carry their query string
        merged = dict(params or {}) if absolute else {**self.params, **(params or {})}
        if merged:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(merged, doseq=True)
        return url

    def get(self, endpoint: str = "", params: dict | None = None):
        """One GET with retries (429 / 5xx / network errors, honouring ``Retry-After``).
        Returns ``(json_body, headers)``."""
        url = self._url(endpoint, params)
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests_made += 1
            req = urllib.request.Request(url, headers=self.headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:     # noqa: S310 - http(s) only
                    raw = resp.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ValueError(f"Response from {redact(url)} exceeds {MAX_RESPONSE_BYTES} bytes")
                    return (json.loads(raw) if raw.strip() else None), dict(resp.headers)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.retries:
                    raise APIError(exc.code, url, body) from None
                delay = exc.headers.get("Retry-After")
                pause = float(delay) if delay and delay.replace(".", "", 1).isdigit() else self.backoff * 2 ** attempt
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.retries:
                    raise ConnectionError(f"Could not reach {redact(url)}: {exc}") from None
                pause = self.backoff * 2 ** attempt
            time.sleep(min(pause, 60.0))
        raise AssertionError("unreachable")

    def fetch(self, endpoint: str = "", *, params: dict | None = None, records: str | None = None,
              pagination: str | None = None, page_param: str = "page", start_page: int = 1,
              size_param: str | None = None, page_size: int | None = None, offset_param: str = "offset",
              limit_param: str = "limit", cursor_param: str = "cursor", cursor_path: str = "next_cursor",
              next_path: str = "next", max_pages: int = 10_000, flatten: bool = True) -> pl.DataFrame:
        """All records from an endpoint as a DataFrame (nested objects flattened to ``a.b`` columns).

        ``records``: dotted path to the list of records in each response (default: the body if
        it's a list, else the first list-valued field)."""
        if pagination not in (None, "page", "offset", "cursor", "next", "link"):
            raise ValueError("pagination must be None, 'page', 'offset', 'cursor', 'next' or 'link'")
        rows, params = [], dict(params or {})
        if pagination == "page":
            params[page_param] = start_page
            if size_param and page_size:
                params[size_param] = page_size
        elif pagination == "offset":
            params[offset_param], params[limit_param] = 0, page_size or 100
        target, pages = endpoint, 0
        while True:
            body, headers = self.get(target, params)
            pages += 1
            items = _dig(body, records) if records else (
                body if isinstance(body, list) else next((v for v in (body or {}).values() if isinstance(v, list)), []))
            if items is None:
                items = []
            if not isinstance(items, list):
                raise ValueError(f"'{records}' in the response is not a list")
            rows.extend(items)
            if pagination is None or not items or pages >= max_pages:
                break
            if pagination == "page":
                if page_size and len(items) < page_size:
                    break
                params[page_param] += 1
            elif pagination == "offset":
                if len(items) < params[limit_param]:
                    break
                params[offset_param] += len(items)
            elif pagination == "cursor":
                cursor = _dig(body, cursor_path)
                if not cursor:
                    break
                params[cursor_param] = cursor
            elif pagination == "next":
                nxt = _dig(body, next_path)
                if not nxt:
                    break
                target, params = urllib.parse.urljoin(self._url(endpoint, None), nxt), {}
            elif pagination == "link":
                m = _LINK_NEXT.search(headers.get("Link") or headers.get("link") or "")
                if not m:
                    break
                target, params = urllib.parse.urljoin(self._url(endpoint, None), m[1]), {}
        if not rows:
            return pl.DataFrame()
        if flatten and all(isinstance(r, dict) for r in rows):
            return pl.json_normalize(rows, separator=".", infer_schema_length=None)
        return pl.DataFrame({"value": rows})

    def read(self, *, endpoint: str = "", **kw) -> pl.DataFrame:
        return self.fetch(endpoint, **kw)

    def __repr__(self) -> str:
        return f"RESTSource({redact(self.base_url)})"
