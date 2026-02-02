---
layout: 'distill'
title: 'A Comprehensive Look at Logging in a Modular MLOps Pipeline'
date: '2025-04-17'
description: 'This article explores how logging is integrated into the pipeline, referencing relevant Python scripts in dependencies/logging_utils, Hydra configs under configs/logging_utils, and how each stage’s logs tie back to DVC and MLflow runs.'
img: 'assets/img/pipeline_worker_female.jpg'
tags: [mlops,logging,logging-strategy,unified-logging,hydra,mlflow,prefect,log-levels,reproducibility]
category: 'MLOps: Designing a Modular Pipeline'
authors: 'Tobias Klein'
comments: true
featured: true
---

<!-- _projects/a-comprehensive-look-at-logging-in-a-modular-mlops-pipeline.md -->
<br>

<!-- documentation/articles/logging/long_post.md -->
# A Comprehensive Look at Logging in a Modular MLOps Pipeline

> **Note**: This article references the academic demonstration version of the pipeline.  
> Some implementation details have been simplified or removed for IP protection.  
> Full implementation available under commercial license.


## Introduction

Logging is essential for traceability, debugging, and performance monitoring within an MLOps pipeline. When multiple components—Hydra configuration management, DVC versioning, Optuna hyperparameter optimization, MLflow experiment tracking, and tools like Prefect—must interoperate, logging becomes the glue that keeps the entire process auditable and reproducible. This article explores how logging is integrated into the pipeline, referencing relevant Python scripts in dependencies/logging_utils, Hydra configs under configs/logging_utils, and how each stage’s logs tie back to DVC and MLflow runs.

---

## 1. Centralized and Configurable Logging

[configs/logging_utils/base.yaml](https://github.com/kletobias/advanced-mlops-demo/tree/main/configs/logging_utils/base.yaml) stores default logging parameters ([log_file_path](https://github.com/kletobias/advanced-mlops-demo/tree/main/configs/logging_utils/base.yaml#L3), [formatting](https://github.com/kletobias/advanced-mlops-demo/tree/main/configs/logging_utils/base.yaml#L4), [level](https://github.com/kletobias/advanced-mlops-demo/tree/main/configs/logging_utils/base.yaml#L5)). By loading these configurations through Hydra, the pipeline ensures every step—whether a data ingestion script or an Optuna trial—follows consistent logging conventions.

### Example: base.yaml:

```yaml
# configs/logging_utils/base.yaml
defaults:
  - paths: base
  - _self_

log_directory_path: "${paths.directories.logs}/pipeline"
log_file_path: "${.log_directory_path}/${now:%Y-%m-%d_%H-%M-%S}.log"
level: 20
formatter: "%(asctime)s %(levelname)s:%(message)s"
```

Tools like [Prefect](https://docs.prefect.io/v3/get-started/index) and [MLflow](https://mlflow.org/docs/latest) also have their own logging mechanisms. By funneling all logs into a single log directory (and optionally shipping them to a centralized system like [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html) or [Splunk](https://docs.splunk.com/Documentation)), the pipeline avoids fragmentation. Meanwhile, environment variables and Hydra overrides let you switch log levels (for example, from INFO to DEBUG) at runtime without altering code.

---

## 2. Integration with DVC and Hydra

Each pipeline step logs relevant events (for example, reading a CSV, computing file hashes, applying transformations). DVC references these logs as artifacts whenever a step is re-run. The script [scripts/universal_step.py](https://github.com/kletobias/advanced-mlops-demo/tree/main/scripts/universal_step.py) initializes logging by calling [dependencies/logging_utils/setup_logging.py](https://github.com/kletobias/advanced-mlops-demo/tree/main/dependencies/logging_utils/setup_logging.py). This ensures consistent logging across data ingestion, cleaning, feature engineering, and model training stages.

DVC's `dvc repro` triggers steps in sequence; the logs from each step get appended or stored in a new timestamped file, depending on your Hydra config. Because each stage is versioned, rolling back to a prior commit recovers both the scripts and the logs at that point in time.

### Example

```log
[2025-03-21 16:40:29,477][dependencies.general.mkdir_if_not_exists][INFO] - Directory exists, skipping creation
./project/logs/pipeline
[2025-03-21 16:40:30,476][dependencies.io.csv_to_dataframe][INFO] - Read ./project/data/v7/v7.csv, created df
[2025-03-21 16:54:21,317][dependencies.transformations.[medical_transform]][INFO] - Done with core transformation: [medical_transform]
[2025-03-21 16:54:21,325][dependencies.general.mkdir_if_not_exists][INFO] - Directory exists, skipping creation
./project/data/v8
[2025-03-21 16:54:24,451][dependencies.io.dataframe_to_csv][INFO] - Exported df to csv using filepath: ./project/data/v8/v8.csv
[2025-03-21 16:54:27,709][dependencies.metadata.compute_file_hash][INFO] - Generated file hash: 0cee898257560d8e67d10b502b136054d5340a30fa1836d59e40cc53cbd45144
[2025-03-21 16:54:27,857][dependencies.metadata.calculate_metadata][INFO] - Generated metadata for file: ./project/data/v8/v8.csv
[2025-03-21 16:54:27,857][dependencies.metadata.calculate_metadata][INFO] - Metadata successfully saved to ./project/data/v8/v8_metadata.json
[2025-03-21 16:54:27,857][__main__][INFO] - Sucessfully executed step: [medical_transform]
```

---

## 3. Logging in Optuna and MLflow

When running hyperparameter tuning with Optuna (for example, [dependencies/modeling/rf_optuna_trial.py](https://github.com/kletobias/advanced-mlops-demo/tree/main/dependencies/modeling/rf_optuna_trial.py)), the pipeline logs each trial’s metrics to MLflow. [dependencies/logging_utils/calculate_and_log_importances_as_artifact.py](https://github.com/kletobias/advanced-mlops-demo/tree/main/dependencies/logging_utils/calculate_and_log_importances_as_artifact.py) demonstrates how model artifacts (such as feature importances) are logged as well.

Each trial also streams standard log messages (for example, “Trial 1 => RMSE=764781.853”) to the console and the .log file, allowing real-time monitoring. By referencing the unique MLflow run ID in logs, it’s straightforward to cross-check experiment results with pipeline events.

---

## 4. Hooking Into Prefect Workflows

Prefect tasks can leverage the built-in prefect.context.get_run_logger() for simpler orchestration, but in many teams, a shared Python logger is used to unify logs across both Prefect tasks and local scripts. By pulling configuration from Hydra at the start of each flow, you can maintain a consistent format, level, and output location. This avoids the pitfall of scattering logs into separate systems.

---

## 5. Common Pitfalls

- Inconsistent Run Identifiers  
  Failing to tie logs to unique run IDs or MLflow experiment tags leads to confusion about which logs pertain to which runs.

- Overly Verbose Logging  
  Dumping entire dataframes or large model objects floods your log system, making it hard to find meaningful events.

- Lack of Structure  
  Using free-form strings instead of structured logging (JSON or key-value pairs) complicates downstream parsing.

- Ignoring Config Changes  
  Neglecting to log Hydra overrides or environment variable changes can destroy reproducibility.

- Forgetting Error Handling  
  Not capturing stack traces or skipping logging on exceptions can hamper debugging.

---

## 6. Critical Aspects to Get Right

- Structured and Level-Based  
  Using log levels (INFO, DEBUG, WARNING, ERROR) consistently. One can consider a JSON format if robust search and filtering capabilities are needed.

- Centralization  
  It is important to centralize logs in a known directory or logging service; Unifying logs from local runs, MLflow, Optuna, Prefect, etc.

- Tie Logs to Pipeline Artifacts  
  It is good practice to ensure each run is associated with a Hydra config, a DVC stage, and an MLflow run ID. This single run ID is the backbone of reproducibility.

- Minimal Noise  
  Using appropriate log levels means only major transformations, hyperparameters, and relevant errors get logged, not every micro-step.

- Retention  
  Decide how long logs remain accessible, especially if your pipeline is regularly re-run. For large volumes, adopt efficient rotation or archiving.

---

## Conclusion

A robust logging strategy is an essential pillar for any MLOps pipeline. By pairing consistent Hydra-based configs with DVC tracking, MLflow experiment logging, and optional Prefect orchestration, teams ensure they capture all details necessary to debug, audit, and refine each stage of the project. Combined with unique run identifiers, structured logging, and a disciplined approach to error handling, this pipeline design yields highly traceable and reproducible machine learning workflows.



## Video: A Comprehensive Look at Logging in a Modular MLOps Pipeline

<iframe width="560" height="315" src="https://www.youtube.com/embed/nDq3mY0Ap7o" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

**© Tobias Klein 2025 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
