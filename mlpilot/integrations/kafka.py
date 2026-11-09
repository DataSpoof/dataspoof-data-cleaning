"""Streaming with Kafka (needs ``confluent-kafka``): consume JSON records in micro-batches,
score them, and produce the predictions to another topic.

>>> src = KafkaSource("orders", bootstrap_servers="kafka:9092", group_id="churn-scorer")
>>> score_stream(model, src, KafkaSink("orders-scored", bootstrap_servers="kafka:9092"))

Offsets are committed only after a batch has been scored and delivered (at-least-once).
"""

from __future__ import annotations

import json
import time

import polars as pl


class KafkaSource:
    def __init__(self, topic: str, *, bootstrap_servers: str, group_id: str, consumer=None, **config):
        if consumer is None:
            try:
                from confluent_kafka import Consumer
            except ImportError as exc:
                raise ImportError("Kafka streaming needs confluent-kafka: pip install confluent-kafka") from exc
            consumer = Consumer({"bootstrap.servers": bootstrap_servers, "group.id": group_id,
                                 "enable.auto.commit": False, "auto.offset.reset": "earliest", **config})
        self.consumer, self.topic = consumer, topic
        self.consumer.subscribe([topic])

    def batches(self, *, max_records: int = 1000, timeout: float = 1.0, max_batches: int | None = None):
        """Yield ``(DataFrame, messages)`` micro-batches; invalid JSON messages are skipped (counted)."""
        done = 0
        self.skipped = 0
        while max_batches is None or done < max_batches:
            msgs = self.consumer.consume(num_messages=max_records, timeout=timeout)
            good, records = [], []
            for m in msgs or []:
                if m.error():
                    continue
                try:
                    rec = json.loads(m.value())
                except (TypeError, ValueError):
                    self.skipped += 1
                    continue
                if isinstance(rec, dict):
                    good.append(m)
                    records.append(rec)
            done += 1
            if records:
                yield pl.DataFrame(records, infer_schema_length=None), good

    def commit(self, messages):
        if messages:
            self.consumer.commit(message=messages[-1], asynchronous=False)

    def close(self):
        self.consumer.close()


class KafkaSink:
    def __init__(self, topic: str, *, bootstrap_servers: str | None = None, producer=None, **config):
        if producer is None:
            try:
                from confluent_kafka import Producer
            except ImportError as exc:
                raise ImportError("Kafka streaming needs confluent-kafka: pip install confluent-kafka") from exc
            producer = Producer({"bootstrap.servers": bootstrap_servers, "enable.idempotence": True, **config})
        self.producer, self.topic = producer, topic

    def write(self, records: list, keys: list | None = None):
        for i, rec in enumerate(records):
            key = None if keys is None or keys[i] is None else str(keys[i]).encode()
            self.producer.produce(self.topic, value=json.dumps(rec, default=str).encode(), key=key)
        if self.producer.flush(30) != 0:
            raise RuntimeError("Kafka producer could not deliver every message")


def score_stream(model, source: KafkaSource, sink, *, key: str | None = None, keep=None, max_records: int = 1000,
                 max_batches: int | None = None) -> dict:
    """Consume -> score -> produce until ``max_batches`` (None = forever). ``sink`` is a
    :class:`KafkaSink` or any callable taking a list of output dicts."""
    from ..deploy.scoring import resolve_model, score
    model = resolve_model(model)
    stats = {"batches": 0, "records": 0, "seconds": 0.0}
    t0 = time.perf_counter()
    for frame, msgs in source.batches(max_records=max_records, max_batches=max_batches):
        keep_cols = [c for c in (keep or ([key] if key else [])) if c in frame.columns]
        out = score(model, frame, keep=keep_cols).to_dicts()
        keys = frame[key].to_list() if key and key in frame.columns else None
        (sink.write(out, keys) if hasattr(sink, "write") else sink(out))
        source.commit(msgs)                                    # only after delivery: at-least-once
        stats["batches"] += 1
        stats["records"] += len(out)
    stats["seconds"] = round(time.perf_counter() - t0, 3)
    stats["skipped"] = getattr(source, "skipped", 0)
    return stats
