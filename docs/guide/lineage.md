# Lineage & catalogs


```python
from automatedcleaning.lineage import dataset
lin = ac.Lineage.from_env()                       # OPENLINEAGE_URL (Marquez, DataHub, Atlan...) and/or a file
with lin.run("train_churn", inputs=[dataset("warehouse.customers", train_df)],
             outputs=[dataset("models/churn/v7")]):
    model = ac.AutoML("churned").fit(train_df)    # START, then COMPLETE (or FAIL with the stack trace)
lin.emit_tracked_run(run)                          # existing tracker runs and dataset versions, too
ac.LineageGraph.from_file("lineage.jsonl").upstream("models/churn/v7")

from automatedcleaning.integrations import aws_glue
aws_glue.write_table(df, "analytics", "customers", "s3://lake/customers/", partition_by=["country"])
aws_glue.read_table("analytics", "customers"); aws_glue.run_job("nightly-etl", {"--date": "2026-11-08"})

from automatedcleaning.integrations.azure_data_factory import DataFactory
DataFactory(sub, "data-rg", "corp-adf").run_pipeline("CopyOrders", {"date": "2026-11-08"})
```

**What the events carry:** OpenLineage events include each dataset's schema, row and null counts,
and versions.

**What was tested against what:**
- **Marquez** (the OpenLineage reference server): real, in Docker. Jobs, run states, schemas and
  the lineage graph all appeared as expected.
- **Glue:** a moto server speaking the real S3 protocol. This covered Parquet, partitions, the
  catalog schema and job runs.
- **Data Factory:** a fake Azure Resource Manager endpoint, testing the REST contract.
