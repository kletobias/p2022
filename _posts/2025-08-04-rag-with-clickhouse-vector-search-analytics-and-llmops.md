---
layout: distill
title: "RAG with ClickHouse: Vector Search, Analytics, and LLMOps"
date: 2025-08-04
description: "Three Birds, One Engine: How ClickHouse Powers Retrieval-Augmented Generation End‑to‑End"
tags:
  ["ClickHouse", "RAG", "Vector Search", "LLMOps", "Uber", "Database", "SQL"]
---

<!-- _posts/2025-08-04-rag-with-clickhouse-vector-search-analytics-and-llmops.md -->
# RAG with ClickHouse: Vector Search, Analytics, and LLMOps

<br>

> **tl;dr**  
> ClickHouse’s new **VECTOR** type and HNSW index turn the OLAP workhorse into a single system that stores embeddings, serves fast similarity search, joins structured filters, and logs every token for evaluation. If your GenAI stack is drowning in specialized databases, consolidating on ClickHouse may cut complexity without sacrificing performance.

## ClickHouse GitHub Repository

<div class="github-card" data-user="ClickHouse" data-repo="ClickHouse" data-width="400" data-theme="default"></div>
<script src="https://cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

## Vector Storage & Retrieval

- Embeddings fit naturally in `Array(Float32)` or the new `VECTOR` type.
- Built‑in distance functions (`L2Distance`, `CosineDistance`, …) mean **no UDFs, no RPC hops**.
- Optional **HNSW** index delivers **10‑100×** speed‑ups and supports **BF16 / int8** quantization to slash RAM by 60 %+.
- Disable the index and ClickHouse parallel‑scans **billions of rows** in place, keeping recall = 1 for cold data.

```sql
SELECT id, text
FROM docs
ORDER BY CosineDistance(embedding, :query_embedding)
LIMIT 5;
```

## Hybrid Search, Analytics & Joins

Because vector functions are plain SQL, you can:

```sql
WITH ann AS (
  SELECT id, score
  FROM docs
  ORDER BY CosineDistance(embedding, :q)         -- ANN
  LIMIT 100
)
SELECT d.text
FROM ann
JOIN docs d USING id
WHERE d.user_id = 42                             -- rich filter
ORDER BY score
```

One query, one round trip—ideal for RAG post‑ranking, safety checks, and faceted search.

## Ingestion & Streaming

- **ClickPipes** and native Kafka / S3 / GCS connectors ingest **millions of rows /s**.
- Compression keeps storage €‑friendly; schema changes are DDL, not migrations.

## Ecosystem Integration

- **LangChain** and **LlamaIndex** adapters (`Clickhouse(...)`) swap in seamlessly for Chroma or FAISS.
- Amazon Bedrock reference demo shows end‑to‑end RAG in pure SQL + LangChain.

## Observability & Evaluation

**LangSmith** migrated from Postgres to ClickHouse to log every token, trace, and metric—proving it can be your telemetry lakehouse and analytics dashboard in one.

## Feature Store Synergy

Under the same cluster:

1. **Offline** joins for training sets (point‑in‑time correctness).
2. **Online** materialized views for low‑latency features.

No extra serving layer required.

## Operational Footprint

| Characteristic | Detail                                                 |
| -------------- | ------------------------------------------------------ |
| Deployment     | Self‑managed binary, BYOC, or EU‑hosted SaaS           |
| Compliance     | Runs in German regions; data stays in‑zone             |
| IaC            | **Terraform provider (June 2025)** + SQL‑only schema   |
| SLOs           | `system.query_log`, `system.metrics` ready for Grafana |

## Limitations & Caveats

- HNSW index is **beta**. Enable with
  ```sql
  SET enable_vector_similarity_index = 1;
  ALTER TABLE docs MATERIALIZE INDEX hnsw_idx;
  ```
- For <10 ms, >5 k QPS _purely in‑memory_ workloads, a niche vector DB may be cheaper.

## Verdict

ClickHouse now spans three RAG layers:

1. **Vector DB** — disk or ANN.
2. **Feature Store** — structured filters & model features.
3. **Telemetry Warehouse** — traces & evals.

If your team already speaks SQL and values open‑source, consolidating on ClickHouse shrinks moving parts from laptop dev to petabyte clusters—at the cost of betting on an index that’s still hardening. For many, that trade‑off is worth it.
