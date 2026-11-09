"""Ingestion & connectors (module 1 of the platform).

* ``read(source, ...)``      files, globs, URLs, S3/GCS/Azure, databases, warehouses, APIs -> DataFrame
* ``ingest(source, ...)``    + type inference, data-contract validation, quality score, lineage
* ``SQLSource``              any SQLAlchemy URL or DB-API connection (bound parameters only)
* ``SnowflakeSource`` / ``BigQuerySource``   native warehouse connectors (Arrow results)
* ``RESTSource``             JSON APIs with auth, pagination, retries and rate limiting
* ``IncrementalLoader``      only new rows (watermark) or new files, with safe commits
* ``infer_types``            text -> numbers / dates / booleans, reusable across batches
"""

from ._util import quote_ident, redact
from .api import APIError, RESTSource
from .core import IngestResult, ingest, read
from .files import detect_format, read_file
from .incremental import Batch, IncrementalLoader
from .schema import InferBooleans, infer_types
from .sql import SQLSource, is_sql_url
from .warehouse import BigQuerySource, SnowflakeSource

__all__ = ["read", "ingest", "IngestResult", "read_file", "detect_format", "SQLSource", "is_sql_url",
           "SnowflakeSource", "BigQuerySource", "RESTSource", "APIError", "IncrementalLoader", "Batch",
           "infer_types", "InferBooleans", "redact", "quote_ident"]
