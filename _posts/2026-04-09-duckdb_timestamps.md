---
layout: distill
title: 'Timestamp 101 - Unit and Precision are not the same'
date: 2026-04-09
description: 'Illustration of timestamp concepts using DuckDB, showing that unit and precision are not the same thing.'
tags: ['timestamps', 'duckdb', 'datetime', 'time-since-epoch', 'dtype']
category: 'data engineering'
comments: true
---
<br>

# Timestamp 101: unit != precision

## Introduction

When working with timestamps, it's crucial to understand the difference between unit and precision.

Unit and precision are not the same thing. A timestamp can be represented in different units (seconds, milliseconds, microseconds, nanoseconds) without changing the underlying precision of the original timestamp data.

Before we dive into the details, let's look at an example using DuckDB to illustrate these concepts.

This query demonstrates how the same timestamp can be represented in different units and how the length of the numeric representation changes accordingly.

```sql
WITH tsn AS (
  SELECT now() AS ts_dt
)
SELECT
  ts_dt, -- the original timestamp
  epoch_ns(ts_dt) AS ts_ns,
  len(CAST(ts_ns AS VARCHAR)) AS ts_ns_length,
  epoch_us(ts_dt) AS ts_us,
  len(CAST(ts_us AS VARCHAR)) AS ts_us_length,
  epoch_ms(ts_dt) AS ts_ms,
  len(CAST(ts_ms AS VARCHAR)) AS ts_ms_length,
  epoch(ts_dt) AS ts_double,
  len(regexp_extract(CAST(ts_double AS VARCHAR), '^\d+')) AS ts_double_length
FROM tsn;
```

## Explanation of the query

- ts_dt is a TIMESTAMPTZ, a timezone-aware timestamp with microsecond precision.

- epoch_ns, epoch_us, epoch_ms, and epoch are all functions that take a timestamp and return a numeric representation of the time since the Unix epoch (1970-01-01 00:00:00 UTC) in different units (nanoseconds, microseconds, milliseconds, and seconds, respectively).

- The length of the string representation of each epoch value is calculated to show how many digits are in each representation.

```json
[
  {
    "ts_dt": "2026-04-09 08:23:32.808211+02",
    "ts_ns": 1775715812808211000,
    "ts_ns_length": 19,
    "ts_us": 1775715812808211,
    "ts_us_length": 16,
    "ts_ms": 1775715812808,
    "ts_ms_length": 13,
    "ts_double": 1775715812.808211,
    "ts_double_length": 10
  }
]
```

## Key takeaways

If you see ts in a column or variable name, read it as a hint, not proof. It often means
“timestamp.” dt usually reads as “date/time.” The actual type and unit matter more than
the substring.

DuckDB is just the illustration here. now() gives you a current timestamp. DuckDB
documents now() as “Current date and time (start of current transaction),” defines
TIMESTAMPTZ as a “Time zone aware timestamp with microsecond precision,” and says the
configured time zone “defaults to the system time zone.” In the sample output
2026-04-09 08:23:32.808211+02, the +02 shows this example is using the local offset,
not UTC.  ￼

In the example, the same instant is exposed as epoch(), epoch_ms(), epoch_us(), and
epoch_ns(). DuckDB documents those as “seconds,” “milliseconds,” “microseconds,” and
“nanoseconds since the epoch,” and it notes that TIMESTAMP WITH TIME ZONE “does not
store time zone information”; it stores the “INT64 number of non-leap microseconds since
the Unix epoch 1970-01-01 00:00:00+00.”  ￼

The subtle but important part: unit is not the same as precision. A 19-digit *_ns value
does not automatically mean nanosecond-precision data. DuckDB’s own examples show
epoch_ns('...123456+00') -> ...123456000 while epoch_us('...123456+00') ->
...123456. Same instant, different unit, microsecond source precision.  ￼

And epoch() is that same value again, just rendered as seconds plus a fractional part.
So 1775715812.808211 means 1775715812 whole seconds and 808211 microseconds.
DuckDB’s date-part docs say epoch returns DOUBLEs, which is convenient for display,
but exact integer units are usually the safer load-bearing representation in pipelines.  ￼

Rule: first verify the type, then the unit, then the real precision. Names help. Types and
functions decide.

## Links
- [Unix time - Wikipedia](https://en.wikipedia.org/wiki/Unix_time)
- [Date Part Functions – DuckDB](https://duckdb.org/docs/current/sql/functions/datepart)
- [Timestamp with Time Zone Functions – DuckDB](https://duckdb.org/docs/current/sql/functions/timestamptz)
