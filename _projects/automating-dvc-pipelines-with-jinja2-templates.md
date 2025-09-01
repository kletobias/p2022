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

> **Note**: This article references the academic demonstration version of the pipeline.  
> Some implementation details have been simplified or removed for IP protection.  
> Full implementation available under commercial license.


## Introduction  

Manually editing [dvc.yaml](https://github.com/kletobias/advanced-mlops-demo/tree/main/dvc.yaml) whenever a transformation is added or a parameter changes can be both tedious and error-prone. To address this challenge, a templating system using Jinja2 was introduced. This article explains how Jinja2 templates are structured, how Python scripts generate the final dvc.yaml, and why this method has significantly improved the maintainability of a large MLOps pipeline.

1. The Problem with Manual YAML  
Each stage in dvc.yaml typically includes a cmd (the only required field), deps, and outs section. A pipeline with 20 transformations might require 20 nearly identical stage definitions. If a parameter needs updating or a script is renamed, each stage’s definition must be edited manually—an error-prone process, particularly under tight deadlines or when collaborating with multiple contributors. In a larger pipeline with dozens of transformations, the overhead grows exponentially.

2. The Jinja2 Solution  
   A Jinja2 template file is set up to resemble a typical dvc.yaml, but placeholders are inserted for all parameters. A Python script, retrieves a list of transformations, then populates the template for each stage.

### Example: Template that generates dvc.yaml

Our template creates all stages + plots in dvc.yaml

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/jinja_template.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 1: Contents of the Jinja2 template. Business logic is kept in .py files only. Note: Image shows part of the template.
</div>

This approach keeps dvc.yaml in sync with the Hydra configurations at all times, so new transformations can be added by adjusting the config rather than editing YAML directly. This eliminates repetitive tasks and ensures that each pipeline stage follows a standardized format.

3. Handling Updates & Diffs  
Overwriting manually edited dvc.yaml files is a concern. To address this, the Prefect generation process includes a diff check. If differences exist, the script prompts the user or logs a warning, ensuring that unintended changes do not go unnoticed. In practice, manual edits are rare because the template captures most scenarios, and unusual requirements are better handled within the Hydra configs themselves.

## Conclusion  

Jinja2 templating has proven invaluable in keeping dvc.yaml concise and up to date. Even as the pipeline expands, the overhead of adding or modifying stages remains small. Templating enforces consistency, encourages best practices, and supports a DRY (Don’t Repeat Yourself) philosophy. Further details about the Jinja2 file and Python script are available in the repository. Future posts will examine how the pipeline logs each step’s output and metadata, integrating seamlessly with tools like Hydra, DVC, and MLflow.



## Video: Automating DVC Pipelines with Jinja2 Templates

<iframe width="560" height="315" src="https://www.youtube.com/embed/LpklKS1aXkw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


