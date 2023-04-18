---
layout: distill
title: 'The Tasks In Every Machine Learning Project:<br>Structured Data'
date: 2023-02-21
description: 'The six tasks in every machine learning project with structured data.'
img: 'assets/img/838338477938@+-398898935.jpg'
tags: ['predictive-modeling', 'hyperparameter-optimization', 'reproducable-code', 'tabular-data', 'feature-engineering']
category: ['tabular-data']
authors: 'Tobias Klein'
comments: true
---
<br>
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#summary">Summary</a></div>
    <div class="no-math"><a href="#task-no-1--define-problem">Task No. 1 | Define Problem</a></div>
    <div class="no-math"><a href="#task-no-2--analyze-data">Task No. 2 | Analyze Data</a></div>
    <div class="no-math"><a href="#task-no-3--evaluate-candidate-estimators">Task No. 3 | Evaluate Candidate Estimators</a></div>
    <div class="no-math"><a href="#task-no-4--feature-engineering">Task No. 4 | Feature Engineering</a></div>
    <div class="no-math"><a href="#task-no-5--improve-results">Task No. 5 | Improve Results</a></div>
    <div class="no-math"><a href="#task-no-6--present-results">Task No. 6 | Present Results</a></div>
  </nav>
</d-contents>

# The Six Tasks In Every Machine Learning Project: Structured Data

## Summary
A predictive modeling machine learning project can be divided into six main
tasks, as described below using Python. These tasks are part of the prototyping
process and are tailored for tabular data.

<!-- You can find the tasks for image and more generally unstructured data in the -->
<!-- article **The Six Tasks In Every ML Project: Unstructured Data** -->
<!-- _projects/steps-unstructured.md. There is no difference between structured -->
<!-- and unstructured data in the tasks described here, only the subtasks within each -->
<!-- task vary. -->

## Task No. 1 | Define Problem

- Understand the fundamentals: Gain a deep understanding of the goals of the
project and the input data, including variables, data structure, and data
gathering methods.
- Define the model's prediction target and evaluation metric.

## Task No. 2 | Analyze Data

- Perform Exploratory Data Analysis (EDA) to understand the raw data, including
unique values, missing data, data types, and data distribution.
- Create visualizations to analyze univariate and bivariate distributions,
correlations, and skewness of variables.

## Task No. 3 | Evaluate Candidate Estimators: Baseline

- Establish baseline scores using regression/classification models with
minimal requirements on input data, such as RandomForestRegressor or
RandomForestClassifier.
- Perform feature selection using `feature_importances_` or `permutation_importance`.
- Handle missing values and do datatype conversions as needed.
- Be mindful of data leakage and use proper train/test splits and cross-validation techniques.

## Task No. 4 | Feature Engineering

- Perform feature engineering to transform variable distributions, consider
categorical embeddings, and consider libraries like automl or TPOT for model
selection.
- Test smaller subsets of independent variables and analyze prediction results.
- Go back and forth between steps listed under Task No. 3 as necessary.

## Task No. 5 | Improve Results

- Design a test harness to select models with the best scores from Task No. 3.
- Customize the training metric, if needed, and use hyperparameter optimization
techniques such as grid search or random search.
- Use proper cross-validation methods and try ensembles of estimators with custom
weights.
- Iterate between tasks 2-4 as needed.

## Task No. 6 | Present Results

- Finalize the model, make predictions, and document the process.
- Present the work and explain how the final solution addresses the problem
defined at the beginning.
- Acknowledge limitations and areas for further improvement.
