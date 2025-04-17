---
layout: 'distill'
title: 'A Comprehensive Look at Modular Code in an MLOps Pipeline'
date: '2024-01-14'
description: 'Modular code refers to designing each pipeline stage—data ingestion, preprocessing, model training, evaluation, and deployment—as a distinct module with well-defined inputs, outputs, and responsibilities.'
img: 'assets/img/pipeline_worker_female.jpg'
tags: [mlops,modular-code,pipeline-design,stage-module,atomic-transformations,scalablility]
category: 'MLOps: Designing a Modular Pipeline'
authors: 'Tobias Klein'
comments: true
---

<br>

# A Comprehensive Look at Modular Code in an MLOps Pipeline

**Introduction**  
Modular code refers to designing each pipeline stage—data ingestion, preprocessing, model training, evaluation, and deployment—as a distinct module with well-defined inputs, outputs, and responsibilities. This principle underpins maintainability, scalability, and straightforward debugging. By separating out each function or transformation step into its own file and configuration, developers ensure that changes remain localized, dependencies stay clear, and new features can be introduced with minimal disruption.

---

## 1. Why Modular Code Matters

1. **Single Responsibility per Module**  
   A script or function should handle a single task. For instance, [dependencies/transformations/mean_profit.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/mean_profit.py) implements profit calculation, while [dependencies/transformations/agg_severities.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/agg_severities.py) focuses on aggregating severities. This prevents large, monolithic scripts that are difficult to test or extend.

2. **Explicit Interfaces**  
   Configuration-driven modules reference typed dataclasses in, for example, [dependencies/transformations/agg_severities.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/agg_severities.py#L10) or [dependencies/modeling/rf_optuna_trial.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/modeling/rf_optuna_trial.py). Each dataclass defines the parameters that feed into a function, forming a clear contract between modules.

3. **Isolation and Testability**  
   When transformations are spread across smaller Python files—[lag_columns.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/lag_columns.py), [drop_rare_drgs.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/drop_rare_drgs.py), etc.—testing becomes simpler because each function can be unit tested with controlled inputs. It is also easier to check logs and outputs when the code path is limited to a single transformation.

4. **Parallel and Distributed Execution**  
   With modules isolated, orchestrators (DVC, Prefect, Airflow) can run steps in parallel if their data dependencies do not overlap. For instance, if one stage aggregates data while another calculates column lags, they can proceed independently before merging results.

---

## 2. Best Practices

- **Keep Configs Separate**  
  Store parameters in dedicated YAML files under [configs/transformations/](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/configs/transformations) or [configs/model_params/](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/configs/model_params). This decouples code from configuration, aligning with Hydra’s compositional approach.

- **Use Clear Naming Conventions**  
  In the transformations folder, each file name (e.g., `mean_profit.py`) matches the YAML config key (e.g., [mean_profit.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/configs/transformations/mean_profit.yaml)). This consistency lowers the barrier to jumping between code and config.

- **Minimize Cross-Module Dependencies**  
  If a module references another transformation’s output, prefer well-defined data structures or intermediary files tracked by DVC. Avoid hidden imports or direct references that can create spaghetti dependencies.

- **Combine Automation & Code Generation**  
  The pipeline uses Jinja2 for generating [dvc.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dvc.yaml) (see [templates_jinja2/long_post.md](../templates_jinja2/long_post.md)) to ensure each transformation stage is automatically listed, removing human error in writing repetitive YAML.

---

## 3. Critical Aspects to Get Right

- **Interface Consistency**  
  Each module should use typed parameters if possible, as done in [dependencies/transformations/rolling_columns.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/rolling_columns.py#L14). This typed approach avoids mismatches.

- **Logging**  
  Centralized logging (see [logging/long_post.md](../logging/long_post.md)) ensures each module’s operations are captured, facilitating troubleshooting.

- **Reuse and Composability**  
  If multiple transformations perform a similar operation—such as computing weighted averages—develop a shared utility in [dependencies/transformations/](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations). This approach prevents code duplication and fosters uniformity.

- **Continuous Integration**  
  Having tests that verify each module’s function ensures new commits do not break existing transformations. DVC stages also keep the pipeline in sync, only re-running modules with changed configs or inputs.

---

## 4. Common Pitfalls

- **Monolithic Scripts**  
  Combining data ingestion, cleaning, feature engineering, and modeling into a single file leads to fragile code. Changes in one step risk breaking everything.

- **Hard-Coded Parameters**  
  Burying hyperparameters or file paths in code prevents easy updates. It can also create inconsistent runs if multiple references exist.

- **Hidden Dependencies**  
  Modules that quietly rely on side effects or global state hamper reproducibility. Each module should cleanly specify its inputs and outputs.

- **Insufficient Testing**  
  Without proper test coverage, incremental changes accumulate technical debt. Testing each transformation individually is a more reliable approach.

---

## Conclusion

The project’s modular design ensures each script has a clear purpose, references a dedicated YAML config, and outputs consistent artifacts tracked by DVC. This organization—rooted in single-responsibility modules and typed interfaces—prevents a variety of MLOps headaches. By combining modular code with robust configuration management, logging, and pipeline orchestration, teams can sustain rapid iteration without losing clarity or reproducibility.



## Video: A Comprehensive Look at Modular Code in an MLOps Pipeline

<iframe width="560" height="315" src="https://www.youtube.com/embed/lGIJrzQ3-q8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


