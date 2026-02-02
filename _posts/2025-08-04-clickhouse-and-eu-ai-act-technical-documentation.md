---
layout: distill
title: "ClickHouse and EU AI Act Technical Documentation"
date: 2025-08-04
description: "Assessing how ClickHouse supports each Annex IV documentation aspect under the EU AI Act and identifying the missing gaps."
tags:
  ["clickhouse", "eu-ai-act", "technical-documentation", "compliance", "rag"]
---

# ClickHouse and EU AI Act Technical Documentation
<br>
> **tl;dr** Assessing how ClickHouse supports each Annex IV documentation aspect under the EU AI Act and identifying the missing gaps.

## ClickHouse GitHub Repository

<div class="github-card" data-user="ClickHouse" data-repo="ClickHouse" data-width="400" data-theme="default"></div>
<script src="https://cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

## Coverage matrix

| Aspect                                 | EU AI Act reference  | ClickHouse support | Notes                                                                                             |
| -------------------------------------- | -------------------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| System description & intended purpose  | Annex IV (a)         | partial            | Store Markdown/PDF blobs or URLs; authoring, version control and approvals in Git/Confluence/DMS. |
| Design & architecture incl. algorithms | Annex IV (b)         | partial            | Metadata tables OK; diagrams and code history still need external SCM.                            |
| Data sets & governance metadata        | Annex IV (c), Art 10 | yes                | Columnar tables track lineage; S3/Azure functions read raw samples; populate via ETL.             |
| Performance metrics & test results     | Annex IV (d)         | yes                | Write eval metrics to MergeTree tables; dashboards via Grafana/CH queries.                        |
| Risk‑management records                | Art 9, Annex IV (e)  | partial            | Risk register can be a table, but workflow, review sign‑off and mitigation docs stay outside.     |
| Human‑oversight description            | Annex IV (f), Art 14 | partial            | Narrative oversight procedures stored as blobs; enforcement lives in Policy/OPS tools.            |
| Post‑market monitoring & incidents     | Art 15               | yes                | High‑volume logs → materialized‑view alerts; retains ≥10 y with tiered storage.                   |
| Automatic event logging                | Art 12               | yes                | system.query_log, system.trace_log, custom tables satisfy lifetime traceability.                  |
| Cyber‑security evidence                | Art 15               | yes                | Store audit logs (ClickHouse Cloud Audit tab) and security events for forensic replay.            |
| Versioning & change history            | Annex IV (h)         | partial            | Retention handled; true diff/merge & signed releases require Git/MLflow/OCI registry.             |

## Key gaps

1. **Authoring & lifecycle management of narrative documents** – use Git‑based docs or DMS.
2. **Electronic signatures / approval workflows** – integrate QMS or e‑sign platform.
3. **Automated generation of the Declaration of Conformity & Annex IV bundle** – templating pipeline (e.g., Jinja + CI) that pulls metrics from ClickHouse and docs from Git.
4. **End‑to‑end lineage across data, model, code** – store hashes/IDs in ClickHouse but track artefacts in MLflow + OpenLineage.

## Recommended compliance stack

- **ClickHouse** – structured logs, metrics, lineage columns.
- **Git + Markdown/Asciidoc** – narrative documents under version control.
- **MLflow or equivalent** – model registry and artefact storage.
- **Quality Management System (QMS)** – risk workflow, approval signatures.
- **CI/CD pipeline** – automate Annex IV package assembly and DoC generation.

## Conclusion

ClickHouse excels at high‑volume logging, metrics, and traceability required by the EU AI Act, but needs complementary tooling for narrative documentation, signatures, and artefact governance. Combine it with Git, MLflow, and a QMS to achieve full compliance while keeping analytics fast and cost‑efficient.

---

**© Tobias Klein 2025 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
