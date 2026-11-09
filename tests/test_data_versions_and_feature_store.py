"""Data version control and the feature store."""

import datetime as dt

import polars as pl
import pytest

import automatedcleaning as ac
from automatedcleaning import DataRepository, Entity, FeatureStore, FeatureView


# ── data version control ────────────────────────────────────────────────────

@pytest.fixture()
def customers():
    return pl.DataFrame({"customer_id": [1, 2, 3, 4], "plan": ["a", "b", "a", "c"], "spend": [10.0, 20.0, 30.0, 40.0]})


def test_commit_checkout_dedup_and_tags(tmp_path, customers):
    repo = DataRepository(tmp_path / "repo")
    v1 = repo.commit(customers, "customers", message="initial", tags=["gold"])
    again = repo.commit(customers.clone(), "customers", message="same data")
    assert again.version == 1 and v1.ref.startswith("customers@v1#")              # identical -> no new version
    v2 = repo.commit(customers.with_columns(pl.col("spend") * 2), "customers", message="doubled", tags=["gold"])
    assert v2.version == 2 and v2.parent == 1
    assert repo.get("customers", tag="gold").version == 2 and "gold" not in repo.get("customers", 1).tags
    assert repo.checkout("customers", 1).equals(customers)
    assert repo.checkout("customers")["spend"].to_list() == [20.0, 40.0, 60.0, 80.0]
    repo.tag("customers", 1, "baseline")
    assert repo.checkout("customers", tag="baseline").equals(customers)
    log = repo.log("customers")
    assert log["version"].to_list() == [2, 1] and log["message"].to_list() == ["doubled", "initial"]
    assert repo.datasets().row(0, named=True)["versions"] == 2
    assert len(list((tmp_path / "repo" / "objects").glob("*.parquet"))) == 2
    with pytest.raises(ValueError):
        repo.commit(customers, "../escape")


def test_diff_with_and_without_key(tmp_path, customers):
    repo = DataRepository(tmp_path / "repo")
    repo.commit(customers, "c")
    new = (customers.filter(pl.col("customer_id") != 4)
           .with_columns(pl.when(pl.col("customer_id") == 2).then(99.0).otherwise(pl.col("spend")).alias("spend"))
           .vstack(pl.DataFrame({"customer_id": [5], "plan": ["b"], "spend": [5.0]}))
           .with_columns(pl.lit(True).alias("vip")))
    repo.commit(new, "c")
    d = repo.diff("c", 1, 2, key="customer_id")
    assert (d.added_rows, d.removed_rows, d.changed_rows, d.unchanged_rows) == (1, 1, 1, 2)
    assert d.changed_columns == {"spend": 1} and d.added_columns == ["vip"] and not d.identical
    nk = repo.diff("c", 1, 2)
    assert nk.added_rows == 2 and nk.removed_rows == 2 and nk.unchanged_rows == 2
    assert "rows:" in d.summary()


def test_tampering_is_detected_and_runs_are_linked(tmp_path, customers):
    repo = DataRepository(tmp_path / "repo")
    tr = ac.Tracker(tmp_path / "runs")
    with tr.start_run("etl") as run:
        v = repo.commit(customers, "customers", run=run, source="s3://raw/customers.csv")
    assert v.run_id == run.id and run.tags["dataset.customers"] == v.ref
    with tr.start_run("train") as run2:
        repo.checkout("customers", run=run2)
    assert any(d["name"] == "customers" for d in run2.datasets)
    obj = tmp_path / "repo" / "objects" / v.file
    other = tmp_path / "x.parquet"
    customers.with_columns(pl.col("spend") + 1).write_parquet(other, compression="zstd")
    obj.write_bytes(other.read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        repo.checkout("customers")


def test_gc_removes_unreferenced_objects(tmp_path, customers):
    repo = DataRepository(tmp_path / "repo")
    repo.commit(customers, "c")
    (tmp_path / "repo" / "objects" / "orphan.parquet").write_bytes(b"x")
    assert repo.gc() == 1 and repo.checkout("c").equals(customers)


# ── feature store ───────────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path):
    fs = FeatureStore(tmp_path / "fs")
    fs.apply(Entity("customer", ["customer_id"]),
             FeatureView("stats", entities=["customer"], features=["orders_30d", "spend_30d"], timestamp="as_of",
                         ttl_days=10, owner="growth-team"))
    snaps = []
    for day, mult in ((1, 1), (5, 2), (9, 3)):
        snaps.append(pl.DataFrame({"customer_id": [1, 2], "as_of": [dt.datetime(2024, 1, day)] * 2,
                                   "orders_30d": [1 * mult, 10 * mult], "spend_30d": [5.0 * mult, 50.0 * mult]}))
    for s in snaps:
        fs.ingest("stats", s)
    return fs


def test_point_in_time_join_never_uses_future_values(store):
    labels = pl.DataFrame({"customer_id": [1, 1, 1, 2, 2, 3],
                           "event_timestamp": [dt.datetime(2024, 1, 3), dt.datetime(2024, 1, 5), dt.datetime(2024, 1, 30),
                                               dt.datetime(2023, 12, 31), dt.datetime(2024, 1, 10), dt.datetime(2024, 1, 6)],
                           "churned": [0, 1, 0, 1, 0, 1]})
    train = store.get_historical_features(labels, ["stats:*"])
    assert train.columns == ["customer_id", "event_timestamp", "churned", "orders_30d", "spend_30d"]
    assert train["orders_30d"].to_list() == [1, 2, None, None, 30, None]
    # Jan 3 -> Jan 1 snapshot (not Jan 5); Jan 5 -> same-day snapshot; Jan 30 -> last snapshot is 21 days old > ttl
    # Dec 31 -> nothing existed yet; customer 3 has no features
    one = store.get_historical_features(labels, ["stats:spend_30d"], full_names=True)
    assert "stats__spend_30d" in one.columns and "orders_30d" not in one.columns
    with pytest.raises(ValueError, match="event_timestamp"):
        store.get_historical_features(labels.drop("event_timestamp"), ["stats:*"])
    with pytest.raises(ValueError, match="no features"):
        store.get_historical_features(labels, ["stats:nope"])


def test_online_store_latest_ttl_and_materialize(store):
    now = dt.datetime(2024, 1, 12, tzinfo=dt.timezone.utc)
    online = store.get_online_features(["stats:*"], [{"customer_id": 2}, {"customer_id": 1}, {"customer_id": 99}], now_ts=now)
    assert online["spend_30d"].to_list() == [150.0, 15.0, None]
    stale = store.get_online_features(["stats:orders_30d"], [{"customer_id": 1}],
                                      now_ts=dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc))
    assert stale["orders_30d"][0] is None
    late = pl.DataFrame({"customer_id": [1], "as_of": [dt.datetime(2024, 1, 2)], "orders_30d": [777], "spend_30d": [1.0]})
    store.ingest("stats", late)                                      # older than what's online: ignored online
    assert store.get_online_features(["stats:orders_30d"], [{"customer_id": 1}], now_ts=now)["orders_30d"][0] == 3
    assert store.materialize("stats", end=dt.datetime(2024, 1, 6)) == 2
    assert store.get_online_features(["stats:orders_30d"], [{"customer_id": 1}], now_ts=now)["orders_30d"][0] == 2
    st = store.feature_stats("stats")
    assert st["rows"] == 7 and st["entities"] == 2
    assert store.list_views().row(0, named=True)["owner"] == "growth-team"


def test_registry_guards(store):
    with pytest.raises(ValueError, match="Unknown entities"):
        store.apply(FeatureView("x", entities=["merchant"], features=["a"], timestamp="t"))
    with pytest.raises(ValueError, match="already has data"):
        store.apply(FeatureView("stats", entities=["customer"], features=["orders_30d"], timestamp="as_of"))
    bad_types = pl.DataFrame({"customer_id": [1], "as_of": [dt.datetime(2024, 2, 1)], "orders_30d": ["many"],
                              "spend_30d": [1.0]})
    with pytest.raises(ValueError, match="Type changes"):
        store.ingest("stats", bad_types)
    with pytest.raises(ValueError, match="Missing columns"):
        store.ingest("stats", pl.DataFrame({"customer_id": [1]}))
    store.apply(FeatureView("stats", entities=["customer"], features=["orders_30d", "spend_30d", "visits"],
                            timestamp="as_of"))                    # adding features is fine
