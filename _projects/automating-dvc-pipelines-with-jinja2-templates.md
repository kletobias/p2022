---
layout: 'distill'
title: 'Automating DVC Pipelines with Jinja2 Templates'
date: '2025-04-17'
description: 'This article explains how Jinja2 templates are structured, how Python scripts generate the final dvc.yaml, and why this method has significantly improved the maintainability of a large MLOps pipeline.'
img: 'assets/img/pipeline_worker_female.jpg'
tags: [mlops,jinja2,advanced-templates,DVC,dvc.yaml,automation,DRY]
category: 'MLOps: Designing a Modular Pipeline'
authors: 'Tobias Klein'
comments: true
---

<!-- _projects/automating-dvc-pipelines-with-jinja2-templates.md -->
<br>

# Automating DVC Pipelines with Jinja2 Templates

Introduction  
Manually editing [dvc.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dvc.yaml) whenever a transformation is added or a parameter changes can be both tedious and error-prone. To address this challenge, a templating system using Jinja2 was introduced. This article explains how Jinja2 templates are structured, how Python scripts generate the final dvc.yaml, and why this method has significantly improved the maintainability of a large MLOps pipeline.

1. The Problem with Manual YAML  
Each stage in dvc.yaml typically includes a cmd (the only required field), deps, and outs section. A pipeline with 20 transformations might require 20 nearly identical stage definitions. If a parameter needs updating or a script is renamed, each stage’s definition must be edited manually—an error-prone process, particularly under tight deadlines or when collaborating with multiple contributors. In a larger pipeline with dozens of transformations, the overhead grows exponentially.

2. The Jinja2 Solution  
   A Jinja2 template file is set up to resemble a typical dvc.yaml, but placeholders are inserted for stage names, commands, dependencies, and outputs. A Python script, generate_dvc_yaml_core.py, retrieves a list of transformations from a Hydra config, then populates the template for each stage. For instance, if a transformation is called “lag_columns,” the script automatically configures cmd: python universal_step.py +transform=lag_columns and generates the relevant deps and outs.

### Example: Jinja2 Template that generates dvc.yaml

The template creates all stages + plots in dvc.yaml

- Template: [templates/dvc/generate_dvc.yaml.j2](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/templates/dvc/generate_dvc.yaml.j2)
- Script: [dependencies/templates/generate_dvc_yaml_core.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dependencies/templates/generate_dvc_yaml_core.py)
- Output: [dvc.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/dvc.yaml)

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/jinja_template.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 1: Contents of the Jinja2 template. Business logic is kept in .py files only.
</div>

This approach keeps dvc.yaml in sync with the Hydra configurations at all times, so new transformations can be added by adjusting the config rather than editing YAML directly. This eliminates repetitive tasks and ensures that each pipeline stage follows a standardized format.

### Example: LuaSnip Snippet For Adding a Stage

This snippet takes advantage of the fact that [configs/pipeline/base.yaml](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/configs/pipeline/base.yaml) (the input file) itself is a hydra config.

```snippet
snippet step_jinja "YAML step for jinja to create dvc.yaml"
  - name: ${1:v0}_${2:download_and_save_data}
    cmd_python: \${cmd_python}
    script: \${universal_step_script}
    overrides: setup.script_base_name=${2} transformations=${2} data_versions.data_version_input=${1} data_versions.data_version_output=${5}
    deps:
      - \${universal_step_script}
      - ./configs/transformations/${2}.yaml
      - ./dependencies/${4:transformations}/${2}.py
      - ./configs/data_versions/${1}.yaml
    outs:
      - ./data/${5:}/${5}.csv
      - ./data/${5}/${5}_metadata.json
```

3. Handling Updates & Diffs  
Overwriting manually edited dvc.yaml files is a concern. To address this, the generation process ([scripts/orchestrate_dvc_flow.py](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/tree/main/scripts/orchestrate_dvc_flow.py)) includes a diff check. Whenever a new dvc.yaml is rendered, it is compared against the existing version. If differences exist, the script prompts the user or logs a warning, ensuring that unintended changes do not go unnoticed. In practice, manual edits are rare because the template captures most scenarios, and unusual requirements are better handled within the Hydra configs themselves.

Conclusion  
Jinja2 templating has proven invaluable in keeping dvc.yaml concise and up to date. Even as the pipeline expands, the overhead of adding or modifying stages remains small. Templating enforces consistency, encourages best practices, and supports a DRY (Don’t Repeat Yourself) philosophy. Further details about the Jinja2 file and Python script are available in the repository. Future posts will examine how the pipeline logs each step’s output and metadata, integrating seamlessly with tools like Hydra, DVC, and MLflow.



## Video: Automating DVC Pipelines with Jinja2 Templates

<iframe width="560" height="315" src="https://www.youtube.com/embed/LpklKS1aXkw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


