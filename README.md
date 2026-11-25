# InDrive Rides at Scale with Apache Spark & Databricks

A hands-on PySpark project exploring how Spark behaves at scale on an **InDrive-style ride-hailing platform**: 2 million rides, driver and zone dimension tables, and deliberately skewed data (one driver takes about 50% of the rides, one zone about 40% of the pickups).

## What I built

The notebook is split into six parts, each ending with `assert` checks that validate the result.

| Part | Topic | What I did |
|------|-------|------------|
| **A** | Partitions | Repartitioned the rides DataFrame into exactly 8 partitions and checked the distribution with `spark_partition_id()`. |
| **B** | Lazy evaluation | Built a transformation plan (rides over 5 km, a `fare_per_km` column, the average per pickup zone) without triggering any action, then executed it with a single `count()`. |
| **C** | Narrow vs wide | Classified common operations (`filter`, `select`, `withColumn`, `groupBy`, `join`, `distinct`) and verified in the physical plans that only the wide one contains an `Exchange` (shuffle). |
| **D** | Joins | Compared a forced sort-merge join (`merge` hint) with a broadcast join on the small driver dimension. Broadcast ran in about 0.60s vs about 1.04s for the shuffle join, and the plans show `SortMergeJoin` vs `BroadcastHashJoin`. |
| **E** | Data skew | Measured the skew after `repartition(8, "driver_id")` (the largest partition held about 69% of the rows), then fixed the aggregation with salting and a two-stage `groupBy`, checking that the counts are identical to the unsalted result. |
| **F** | Delta Lake & windows | Wrote the driver dimension as a Delta table, ran a `MERGE` to update one driver's rating (4.9 to 4.95), and used time travel (`VERSION AS OF 0`) to confirm the old value is preserved. Also computed daily earnings per driver, a running total, and a daily driver ranking with window functions. |

## Repository content

```
indrive_spark/
├── exercise_indrive_spark.py     # Databricks notebook (source format)
├── exercise_indrive_spark.html   # Exported notebook with outputs, proof that all asserts pass
└── README.md
```

## How to run

**On Databricks (recommended):** import `exercise_indrive_spark.py` as a notebook in Databricks Free Edition (serverless) and run the cells in order. The setup cell generates the dataset.

**Locally:** the setup cell falls back to a local `SparkSession` when it is not running on Databricks. You need `pyspark` and `delta-spark` installed. You can reduce the dataset size with the `N_ROWS` environment variable.

## Tools

PySpark, Databricks Free Edition (serverless), Delta Lake, Spark SQL window functions.
