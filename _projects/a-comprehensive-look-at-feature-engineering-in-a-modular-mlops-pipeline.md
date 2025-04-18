---
layout: 'distill'
title: 'A Comprehensive Look at Feature Engineering in a Modular MLOps Pipeline'
date: '2025-04-17'
description: 'Effective feature engineering is central to building high-performing and maintainable machine learning pipelines. This project’s codebase illustrates several practices that keep transformations organized, version-controlled, and consistent across training and inference stages.'
img: 'assets/img/pipeline_worker_female.jpg'
tags: [mlops,feature-engineering,atomic-transformations,hydra,dvc,structured-configs,modular-code,separation-of-concerns,version-agnostic-transformations]
category: 'MLOps: Designing a Modular Pipeline'
authors: 'Tobias Klein'
comments: true
---

<br>

# A Comprehensive Look at Feature Engineering in a Modular MLOps Pipeline

**Introduction**  
Effective feature engineering is central to building high-performing and maintainable machine learning pipelines. This project’s codebase illustrates several practices that keep transformations organized, version-controlled, and consistent across training and inference stages. By splitting transformations into distinct Python modules, referencing them via standardized YAML configs, and orchestrating the entire pipeline with DVC, the overall design ensures reproducibility, scalability, and clear data lineage.

---

## 1. Versioning and Lineage

Tracking data lineage and transformation code in the same repository is essential for reproducibility. The script [scripts/universal_step.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/scripts/universal_step.py) invokes specific transformations (for example, [dependencies/transformations/lag_columns.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/lag_columns.py)) based on the Hydra config. Each transformation is defined in both a Python script and a YAML file (e.g., [configs/transformations/lag_columns.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/configs/transformations/lag_columns.yaml)), and all changes to these files are recorded in version control. Additionally, DVC ensures that each input and output artifact is tied to the version of the code that produced it. This setup allows any stage in the pipeline to be reproduced precisely, even months after initial development.

---

## 2. Modular Design

The pipeline divides transformations into atomic steps, each handling a discrete operation such as dropping rare rows, shifting columns, or aggregating severity data. In [scripts/universal_step.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/scripts/universal_step.py), a dictionary called [`TRANSFORMATIONS`](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/scripts/universal_step.py#L74-L147) maps each transformation name (e.g., `drop_rare_drgs`, `agg_severities`) to its corresponding Python function, and `dataclass`. This allows new transformations to be added with minimal impact on the rest of the codebase. For instance, the transformation `lag_columns` only concerns itself with shifting numeric columns; if that logic changes, it does not affect other operations (like `drop_description_columns`).

### Examples

#### Example: Transformation drop_description_columns

[dependencies/logging_utils/log_function_call.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/logging_utils/log_function_call.py)
[dependencies/transformations/drop_description_columns.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/drop_description_columns.py)

```python
from dependencies.logging_utils.log_function_call import log_function_call
from dependencies.transformations.drop_description_columns import (
    DropDescriptionColumnsConfig, # Config for drop_description_columns
    drop_description_columns, # atomic transformation function
)

TRANSFORMATIONS = {
    "drop_description_columns": {
        "transform": log_function_call(drop_description_columns),
        "Config": DropDescriptionColumnsConfig,
    },
    # ...
}
```

#### Example: Transformation agg_severities

[dependencies/logging_utils/log_function_call.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/logging_utils/log_function_call.py)
[dependencies/transformations/agg_severities.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/agg_severities.py)

```python
from dependencies.transformations.agg_severities import (
    AggSeveritiesConfig, # Config for agg_severities transformation
    agg_severities, # atomic transformation function
)

TRANSFORMATIONS = {
    # ...
    "agg_severities": {
        "transform": log_function_call(agg_severities),
        "Config": AggSeveritiesConfig,
    },
    # ...
}
```

---

## 3. Consistency Across Training and Inference

Maintaining identical transformation code for training and inference prevents data drift and performance degradation. The transformations under [dependencies/transformations](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations) can be called by any script that runs in production or testing. By referencing the same YAML configurations (for instance, [configs/transformations/mean_profit.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/configs/transformations/mean_profit.yaml)) during both training and deployment, the project ensures that incoming data is transformed with identical parameters. This consistency is crucial for stable model performance over time.

---

## 4. Scalability and Parallelism

The modular nature of each transformation enables parallel processing. A transformation like `agg_severities` (see [dependencies/transformations/agg_severities.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/agg_severities.py)) can be scaled by distributing the grouped operations over a cluster if necessary. The pipeline design does not currently show an explicit Spark or Dask integration, but the separated data ingestion and transformation scripts are amenable to parallel frameworks. Each step is clearly defined, so adapting it to a distributed context typically involves small code changes.

---

## 5. Data Validation and Monitoring

The pipeline’s design includes typed dataclasses (e.g., [dataclass block for agg_severities.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/agg_severities.py#L10-L21)) to enforce correct parameter types. Logging statements in each transformation (e.g., `"Done with core transformation: drop_rare_drgs"`) also facilitate monitoring. Integrating distribution checks or anomaly detection would be straightforward to implement within each transformation script. Such checks can quickly alert maintainers if an unexpected data shift occurs.

### Example

Relevant lines are highlighted by `>` symbol at the start of the line.

```log
[2025-03-21 16:40:29,477][dependencies.general.mkdir_if_not_exists][INFO] - Directory exists, skipping creation
/Users/tobias/.local/projects/portfolio_medical_drg_ny/logs/pipeline
[2025-03-21 16:40:30,476][dependencies.io.csv_to_dataframe][INFO] - Read /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v7/v7.csv, created df
> [2025-03-21 16:54:21,317][dependencies.transformations.agg_severities][INFO] - Done with core transformation: agg_severities
[2025-03-21 16:54:21,325][dependencies.general.mkdir_if_not_exists][INFO] - Directory exists, skipping creation
/Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8
[2025-03-21 16:54:24,451][dependencies.io.dataframe_to_csv][INFO] - Exported df to csv using filepath: /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8/v8.csv
[2025-03-21 16:54:27,709][dependencies.metadata.compute_file_hash][INFO] - Generated file hash: 0cee898257560d8e67d10b502b136054d5340a30fa1836d59e40cc53cbd45144
[2025-03-21 16:54:27,857][dependencies.metadata.calculate_metadata][INFO] - Generated metadata for file: /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8/v8.csv
[2025-03-21 16:54:27,857][dependencies.metadata.calculate_metadata][INFO] - Metadata successfully saved to /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8/v8_metadata.json
> [2025-03-21 16:54:27,857][__main__][INFO] - Sucessfully executed step: agg_severities
```

---

## 6. Metadata Capture

Metadata is generated for each pipeline stage by calling `calculate_and_save_metadata` in [scripts/universal_step.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/scripts/universal_step.py). This function records essential properties, such as input shape or timestamp, to the relevant metadata file. Over time, these logs allow auditing of how each feature was computed, including the parameter configuration used at the time of transformation.

### Example

Relevant lines are highlighted by `>` symbol at the start of the line.

```log
[2025-03-21 16:40:29,477][dependencies.general.mkdir_if_not_exists][INFO] - Directory exists, skipping creation
/Users/tobias/.local/projects/portfolio_medical_drg_ny/logs/pipeline
[2025-03-21 16:40:30,476][dependencies.io.csv_to_dataframe][INFO] - Read /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v7/v7.csv, created df
[2025-03-21 16:54:21,317][dependencies.transformations.agg_severities][INFO] - Done with core transformation: agg_severities
[2025-03-21 16:54:21,325][dependencies.general.mkdir_if_not_exists][INFO] - Directory exists, skipping creation
/Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8
[2025-03-21 16:54:24,451][dependencies.io.dataframe_to_csv][INFO] - Exported df to csv using filepath: /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8/v8.csv
> [2025-03-21 16:54:27,709][dependencies.metadata.compute_file_hash][INFO] - Generated file hash: 0cee898257560d8e67d10b502b136054d5340a30fa1836d59e40cc53cbd45144
> [2025-03-21 16:54:27,857][dependencies.metadata.calculate_metadata][INFO] - Generated metadata for file: /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8/v8.csv
> [2025-03-21 16:54:27,857][dependencies.metadata.calculate_metadata][INFO] - Metadata successfully saved to /Users/tobias/.local/projects/portfolio_medical_drg_ny/data/v8/v8_metadata.json
[2025-03-21 16:54:27,857][__main__][INFO] - Sucessfully executed step: agg_severities
```

---

## 7. Clear Separation of Concerns

Distinct scripts handle ingestion ([dependencies/ingestion/download_and_save_data.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/ingestion/download_and_save_data.py)), cleaning ([dependencies/cleaning/sanitize_column_names.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/cleaning/sanitize_column_names.py)), and feature engineering ([dependencies/transformations/mean_profit.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/transformations/mean_profit.py), etc.). This prevents confusion over which part of the code deals with each phase of the pipeline. Moreover, each script references a separate YAML config in the `configs/transformations/` folder, making it simple to locate or modify specific functionality without breaking other parts of the pipeline.

### Example: Transformation mean_profit

[configs/transformations/mean_profit.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/configs/transformations/mean_profit.yaml)

```yaml
defaults:
  - base
  - _self_

mean_profit:
  mean_profit_col_name: mean_profit
  mean_charge_col_name: mean_charge
  mean_cost_col_name: mean_costly
```

---

## Conclusion

The feature engineering flow in this project reflects many MLOps best practices: transformations are version-controlled, modular, and consistent. Logging, metadata capture, and typed configurations also strengthen reliability. These design choices reduce technical debt, accelerate development, and provide a clear record of how each feature was produced. As models evolve or new transformations are added, the underlying structure remains resilient, ensuring minimal risk of pipeline failures and guaranteeing reproducibility.



## Video: A Comprehensive Look at Feature Engineering in a Modular MLOps Pipeline

<iframe width="560" height="315" src="https://www.youtube.com/embed/zWC_Y7ei0kk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


