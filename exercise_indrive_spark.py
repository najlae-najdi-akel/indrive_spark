# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Exercise -- InDrive rides at scale
# MAGIC
# MAGIC Same Spark concepts as the BIM lab, a new business. Run the setup cell, then complete every `TODO`. Each part ends
# MAGIC with `assert` checks -- when they pass, your code is right. Answer the numbered questions (Q1-Q6) in your README.md.
# MAGIC
# MAGIC Databricks Free Edition is serverless: there is no `df.rdd`, no `sparkContext`, and no classic Spark UI, which is why
# MAGIC the helpers use `spark_partition_id()` and `plan_text()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup (given -- do not edit)

# COMMAND ----------

import os
import time
from contextlib import redirect_stdout
from io import StringIO

from pyspark.sql import functions as F
from pyspark.sql.window import Window

try:
    spark
    ON_DATABRICKS = True
except NameError:
    ON_DATABRICKS = False
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master("local[4]").appName("indrive-exercise")
        .config("spark.sql.shuffle.partitions", "8")
    )
    builder = builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension").config(
        "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    )
    if os.environ.get("DELTA_JARS"):
        # jars baked into the Docker image (works offline / behind a proxy)
        spark = builder.config("spark.jars", os.environ["DELTA_JARS"]).getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip  # pip install delta-spark

        spark = configure_spark_with_delta_pip(builder).getOrCreate()

    def display(df, n=20):
        df.show(n, truncate=False)


N_ROWS = int(os.environ.get("N_ROWS", 2_000_000))


def timed(label, fn):
    t0 = time.time()
    result = fn()
    print(f"{label}: {time.time() - t0:.2f}s")
    return result


def partition_sizes(df):
    return df.groupBy(F.spark_partition_id().alias("partition")).count().orderBy("partition")


def plan_text(df):
    """The physical plan as a string, so your asserts can look for 'Exchange', 'BroadcastHashJoin'..."""
    buf = StringIO()
    with redirect_stdout(buf):
        df.explain()
    return buf.getvalue()


drivers_rows = [(1, 'Youssef Amrani', 'Casablanca', 4.8), (2, 'Salma Idrissi', 'Casablanca', 4.6), (3, 'Karim Bennis', 'Casablanca', 4.9), (4, 'Nadia El Fassi', 'Casablanca', 4.5), (5, 'Hamza Ouazzani', 'Casablanca', 4.7), (6, 'Imane Tazi', 'Casablanca', 4.4), (7, 'Rachid Lahlou', 'Casablanca', 4.9), (8, 'Zineb Chraibi', 'Casablanca', 4.6)]
zones_rows = [(1, 'Maarif', 'Casablanca'), (2, 'Ain Diab', 'Casablanca'), (3, 'Sidi Belyout', 'Casablanca'), (4, 'Hay Hassani', 'Casablanca'), (5, 'Bourgogne', 'Casablanca'), (6, 'Oulfa', 'Casablanca'), (7, 'Sidi Moumen', 'Casablanca'), (8, 'Anfa', 'Casablanca')]
dim_driver = spark.createDataFrame(drivers_rows, "driver_id INT, driver_name STRING, city STRING, rating DOUBLE")
dim_zone = spark.createDataFrame(zones_rows, "zone_id INT, zone_name STRING, city STRING")

# InDrive rides at scale. Deliberately skewed: driver 3 takes ~50% of rides, zone 1 is ~40% of pickups.
rides = (
    spark.range(N_ROWS)
    .withColumn("ride_id", F.col("id"))
    .withColumn("driver_id", F.when(F.rand(1) < 0.5, F.lit(3)).otherwise((F.rand(2) * 8 + 1).cast("int")))
    .withColumn("pickup_zone_id", F.when(F.rand(3) < 0.4, F.lit(1)).otherwise((F.rand(4) * 8 + 1).cast("int")))
    .withColumn("distance_km", F.round(F.rand(5) * 12 + 1.5, 1))
    .withColumn("fare_amount", F.round(F.col("distance_km") * (F.rand(6) * 3 + 4) + 10, 2))
    .withColumn("ride_date", F.expr("date_add(DATE'2026-09-01', cast(rand(7) * 14 as int))"))
    .drop("id")
)
print("Spark", spark.version, "| rows:", f"{N_ROWS:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A -- Partitions (slide 6)
# MAGIC
# MAGIC **TODO A1:** create `rides_8`, `rides` spread over exactly 8 partitions.

# COMMAND ----------

rides_8 = rides.repartition(8)  # TODO A1

# COMMAND ----------

assert rides_8 is not None, "A1: replace the TODO -- rides_8 is still None"
print(partition_sizes(rides_8).count(), "partitions")
assert partition_sizes(rides_8).count() == 8, "A1: rides_8 must have exactly 8 partitions"

# COMMAND ----------

# MAGIC %md
# MAGIC **Q1:** why 8 and not 200? How many partitions per CPU core would you aim for, and what goes wrong with too few or too many?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B -- Lazy evaluation (slides 10-11)
# MAGIC
# MAGIC **TODO B1:** build `plan`: rides longer than 5 km, add a `fare_per_km` column, then the average `fare_per_km` per
# MAGIC `pickup_zone_id`. **Do not call any action.**

# COMMAND ----------

plan = (rides
        .filter(F.col("distance_km") > 5)
        .withColumn("fare_per_km", F.col("fare_amount") / F.col("distance_km"))
        .groupBy("pickup_zone_id")
        .agg(F.avg("fare_per_km").alias("avg_fare_per_km")))  # TODO B1

# COMMAND ----------

assert plan is not None, "B1: replace the TODO -- plan is still None"
t0 = time.time()
n_zones = plan.count()          # the action
print(f"action took {time.time() - t0:.2f}s")
assert n_zones == 8, "B1: expected one row per pickup zone (8)"

# COMMAND ----------

# MAGIC %md
# MAGIC **Q2:** which line actually ran the Spark job, and why did building `plan` take almost no time?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C -- Narrow vs wide (slides 13-14)
# MAGIC
# MAGIC **TODO C1:** classify each operation as `"narrow"` or `"wide"` in the `classification` dict.
# MAGIC **TODO C2:** build `wide_df` (a `groupBy` on `driver_id`) and `narrow_df` (a `filter`), so the asserts below can find
# MAGIC (or not find) the shuffle (`Exchange`) in their plans.

# COMMAND ----------

classification = {
    "filter": "narrow",      # TODO C1 -- "narrow" or "wide"
    "select": "narrow",
    "withColumn": "narrow",
    "groupBy": "wide",
    "join (no broadcast)": "wide",
    "distinct": "wide",
}
wide_df = rides.groupBy("driver_id").count()   # TODO C2
narrow_df = rides.filter("distance_km >5")  # TODO C2

# COMMAND ----------

assert wide_df is not None and narrow_df is not None, "C2: replace the TODO -- wide_df / narrow_df are still None"
expected = {"filter": "narrow", "select": "narrow", "withColumn": "narrow", "groupBy": "wide", "join (no broadcast)": "wide", "distinct": "wide"}
assert classification == expected, "C1: check your classification"
assert "Exchange" in plan_text(wide_df), "C2: the wide plan must contain an Exchange (shuffle)"
assert "Exchange" not in plan_text(narrow_df), "C2: the narrow plan must NOT contain an Exchange"
print("C: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D -- Joins (slide 15)
# MAGIC
# MAGIC **TODO D1:** `shuffle_join` -- rides joined to `dim_driver` on `driver_id`, forcing a sort-merge join with a hint.
# MAGIC **TODO D2:** `broadcast_join` -- the same join, broadcasting the small side.

# COMMAND ----------

shuffle_join = rides.join(dim_driver.hint("merge"), "driver_id")   # TODO D1
broadcast_join = rides.join(F.broadcast(dim_driver), "driver_id")  # TODO D2

# COMMAND ----------

assert shuffle_join is not None and broadcast_join is not None, "D1/D2: replace the TODOs -- the joins are still None"
assert "SortMergeJoin" in plan_text(shuffle_join), "D1: expected a SortMergeJoin"
assert "BroadcastHashJoin" in plan_text(broadcast_join), "D2: expected a BroadcastHashJoin"
timed("shuffle join   ", lambda: shuffle_join.groupBy("city").count().collect())
timed("broadcast join ", lambda: broadcast_join.groupBy("city").count().collect())

# COMMAND ----------

# MAGIC %md
# MAGIC **Q3:** why does the broadcast version avoid the shuffle of the big table, and what limits how big the broadcast side can be?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part E -- Skew (slides 7, 16-17)
# MAGIC
# MAGIC **TODO E1:** compute `hot_share`: the fraction of all rows held by the largest partition after
# MAGIC `rides.repartition(8, "driver_id")`.
# MAGIC **TODO E2:** `salted_counts` -- rides per driver computed with salting (8 salts, two-stage aggregation), ordered by
# MAGIC `driver_id`, as a list of `(driver_id, count)` tuples.

# COMMAND ----------

sizes = partition_sizes(rides.repartition(8, "driver_id")).collect()
total = sum(r["count"] for r in sizes)
hot_share = max(r["count"] for r in sizes) / total      # TODO E1
salted = rides.withColumn("salt", (F.rand(7) * 8).cast("int"))
stage1 = salted.groupBy("driver_id", "salt").count()
stage2 = (stage1.groupBy("driver_id")
                .agg(F.sum("count").alias("count"))
                .orderBy("driver_id"))
salted_counts = [(r["driver_id"], r["count"]) for r in stage2.collect()]   # TODO E2

# COMMAND ----------

assert hot_share is not None and salted_counts is not None, "E1/E2: replace the TODOs -- hot_share / salted_counts are still None"
print(f"largest partition holds {hot_share:.0%} of the rows")
assert hot_share > 0.4, "E1: driver 3 should dominate one partition"
plain_counts = [(r["driver_id"], r["count"]) for r in rides.groupBy("driver_id").count().orderBy("driver_id").collect()]
assert salted_counts == plain_counts, "E2: salting must not change the counts"
print("E: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC **Q4:** where would you see this skew in the Spark UI (or the query profile on serverless), and what would you look for?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part F -- Delta and window functions (slides 12, 19)
# MAGIC
# MAGIC **TODO F1:** write `dim_driver` as the Delta table `indrive_dim_driver`, then `MERGE` a rating change for driver 3
# MAGIC (new rating 4.95). **TODO F2:** `running` -- per driver and day: `daily_earnings`, a `running_total` per driver, and
# MAGIC `driver_rank_that_day` (rank of the driver's earnings within each day).

# COMMAND ----------

# TODO F1: write the Delta table, then MERGE the rating change (driver 3 -> 4.95)
spark.sql("DROP TABLE IF EXISTS indrive_dim_driver")   # pour pouvoir relancer la cellule
dim_driver.write.format("delta").mode("overwrite").saveAsTable("indrive_dim_driver")  # version 0

updates = spark.createDataFrame([(3, 4.95)], "driver_id INT, new_rating DOUBLE")
updates.createOrReplaceTempView("rating_updates")

spark.sql("""
    MERGE INTO indrive_dim_driver t
    USING rating_updates s
    ON t.driver_id = s.driver_id
    WHEN MATCHED THEN UPDATE SET t.rating = s.new_rating
""")

daily = (rides.groupBy("driver_id", "ride_date")
              .agg(F.sum("fare_amount").alias("daily_earnings")))

w_running = Window.partitionBy("driver_id").orderBy("ride_date")
w_rank = Window.partitionBy("ride_date").orderBy(F.desc("daily_earnings"))


running = (daily
           .withColumn("running_total", F.sum("daily_earnings").over(w_running))
           .withColumn("driver_rank_that_day", F.rank().over(w_rank)))  # TODO F2

# COMMAND ----------

assert running is not None, "F2: replace the TODO -- running is still None"
now = spark.table("indrive_dim_driver").filter("driver_id = 3").first()["rating"]
before = spark.sql("SELECT rating FROM indrive_dim_driver VERSION AS OF 0 WHERE driver_id = 3").first()["rating"]
print(f"driver 3 rating: version 0 = {before}, now = {now}")
assert now == 4.95 and before != now, "F1: the MERGE must change driver 3 and version 0 must still show the old rating"
assert running.count() == 8 * 14, "F2: one row per (driver, day) -- 8 drivers x 14 days"
assert set(running.columns) >= {"daily_earnings", "running_total", "driver_rank_that_day"}
print("F: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC **Q5:** `running` is used twice below in your own analysis -- would `cache()` help here, and why might it not be
# MAGIC available to you?
# MAGIC
# MAGIC **Q6:** which of these steps would you schedule as a Databricks Job, and which are just exploration?