---
layout: 'distill'
title: 'The Integration Of MLflow In This Project'
date: '2025-04-17'
description: 'MLflow is central to this project’s experiment tracking, artifact management, and reproducible model development. It is integrated through Hydra configurations, S3 synchronization scripts, and Python modeling code that leverages MLflow’s Pythonic API.'
img: 'assets/img/pipeline_worker_female.jpg'
tags: [mlops,mlflow,artifact-tracking,log-metrics,log-parameters,experiment-tracking]
category: 'MLOps: Designing a Modular Pipeline'
authors: 'Tobias Klein'
comments: true
---

<br>

# The Integration Of MLflow In This Project

MLflow is central to this project’s experiment tracking, artifact management, and reproducible model development. It is integrated through Hydra configurations, S3 synchronization scripts, and Python modeling code that leverages MLflow’s Pythonic API.

## Reasons for Using MLflow

MLflow unifies local experimentation and remote artifact sharing. It supports parameter logging, metric tracking, and model registry for consistent collaboration. Language-agnostic design permits flexible usage in Python-based workflows.

## Implementation Details

All MLflow runs store metadata in local mlruns/ directories, which are then synchronized to S3. Configuration YAML files (push_mlruns_s3.yaml and pull_mlruns_s3.yaml) in configs/utility_functions define the AWS bucket name, prefix, local tracking directory, and the exact sync commands. Below is an excerpt from push_mlruns_s3.py showing how the CLI command is executed:

```python
if replace_remote:
    logger.info("Remote objects not present locally will be deleted from S3.")
    confirm = input("Proceed with remote deletion? [y/N]: ")
    if confirm.lower() not in ("y", "yes"):
        logger.info("Aborted by user.")
        return
    sync_command += ["--delete"]

subprocess.run(sync_command, check=True)
logger.info("Push complete.")
```

The local directory path and remote URI both come from Hydra-based YAML entries, ensuring every collaborator uses consistent settings. The scripts in dependencies/io shield developers from manual synchronization procedures.

Experiment Tracking and Model Runs
The modules [dependencies/modeling/rf_optuna_trial.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/modeling/rf_optuna_trial.py) and [dependencies/modeling/ridge_optuna_trial.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/modeling/ridge_optuna_trial.py) demonstrate how MLflow is set up:

```python
mlflow.set_tracking_uri("file:./mlruns")
existing = mlflow.get_experiment_by_name(experiment_name)
if existing is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
...
with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
    mlflow.log_params(final_params)
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
```

Each run is named `trial_X` or `final_model`, storing critical information like RMSE, R², and hyperparameters. Final model runs also log environment details via [mlflow.sklearn.log_model](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/modeling/rf_optuna_trial.py#L173).

## Pitfalls Addressed

- Naming Conventions: The config-driven approach ensures consistent experiment and run names, preventing confusion in multi-user teams.
- Artifact Overload: Only top feature importances and essential CSV files are logged. High-volume data is excluded to keep storage manageable.
- Reproducibility: Final models are logged alongside environment specs, ensuring that any user can re-run the same experiment under identical conditions.
- Security: The S3 URI, bucket name, and prefix are stored in dedicated YAML files. Team members rely on a standard pipeline, minimizing the risk of misconfigurations.

## Comparison with Other Tools

MLflow is more lightweight to maintain compared to large Kubernetes-based platforms (e.g., Kubeflow) or specialized SaaS solutions. It integrates seamlessly with Python scripts and Hydra, which supports complex configuration hierarchies, making it ideal for iterative experiment pipelines in this project.

⸻



## Video: The Integration Of MLflow In This Project

<iframe width="560" height="315" src="https://www.youtube.com/embed/M1WEOWW_9CM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

**© Tobias Klein 2025 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
