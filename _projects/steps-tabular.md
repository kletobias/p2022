---
layout: distill
title: 'The Tasks In Every Machine Learning Project:<br>Tabular Data'
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

# The Six Tasks In Every Machine Learning Project: Tabular Data

## Summary
A predictive modeling machine learning project can be divided into six main
tasks. The tasks listed below are part of the *prototyping* process (as
described using Python), not the production process, which is often handled
using a *faster* implementation of the final model written in *C++* for example.
The following list is tailored for tabular data.
<!-- You can find the tasks for image and more generally unstructured data in the -->
<!-- article **The Six Tasks In Every ML Project: Unstructured Data** -->
<!-- _projects/steps-unstructured.md. There is no difference between structured -->
<!-- and unstructured data in the tasks described here, only the subtasks within each -->
<!-- task vary. -->

## Task No. 1 | Define Problem
1. Understand the fundamentals: Understand and characterize the problem.
    - *Goal: Get a better understanding of the goals of the project.*
    - Understand the input data. E.g., 
    - What are the independent and dependent variables?
    - What do the values (rows) in each of the columns in the dataset
        represent?
    - What is the structure and size of the train/test dataset?
    - How was the input data gathered?
    - In which form is the input data given? (flat file, database table, etc.)
    - What is it, that the model has to predict in the end?
    - Evaluation metric:
        - What is the Evaluation metric?
        - Or what are the candidates for the evaluation metric, given the problem?

## Task No. 2 | Analyze Data

- [Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
    - Analyze the how the raw data looks like. E.g.,
        - Look at unique values for each column. 
        - Visualize missing data
        - Analyze the data type(s) for each variable in the input data.
        - See, if the given data fulfills the requirements of *tidy data*.
            - Each column is a single variable.
            - Each row represents exactly one sample (also called: record or instance).
    - Create Visualizations that show:
        - Univariate/bivariate distributions between variables.
        - Correlations between independent variables.
        - Review the skew of the distributions of each variable.

## Task No. 3 | Evaluate Candidate Estimators

- Baseline Scores
    - Base Line Scores
        - In general: Be mindful of possible data leakage, adhere to best practices
            using adequate train/test splits function and cross-validation
            techniques.
        - The following steps depend on the type of model and on the findings of the
            second step Links lead to the [*scikit-learn*](https://scikit-learn.org/stable/modules/classes.html) implementation of the mentioned tools.
        - Choose a regression/classification model depending on the problem.
        - Fill missing values using a fitting fill strategy.
        - Do datatype conversions where necessary. E.g., Convert categorical
            data given as string datatype to integer datatype.
        - Get a first benchmark using a model, that has little requirements on the
            input data that can be interpreted well. E.g., [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor) or [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
            - Start with feature selection, using [*feature\_importances\_*](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.feature_importances_) or in the
                case of high cardinality features possibly [*permutation\_importance*](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance).

## Task No. 4 | Feature Engineering

- Feature Engineering & Evaluation Of Models
    - Transform distributions of the variables, where necessary. E.g., log
        transform, binning, normalizing, creating different *views* of the data.
    - Consider categorical embeddings, especially if using a deep learning
        model, that can utilize the added information.
    - Consider libraries like [*automl*](https://github.com/automl) or [*TPOT*](http://epistasislab.github.io/tpot/),
        to help with model selection. Results can depend on number
        of iterations / time given for the algorithm to create pipelines
        (A Cuda GPU can be helpful).
    - Test smaller subsets of the independent variables.
    - Analyze the prediction results and go back and forth between steps
        listed under *Task No. 3*, if necessary.

## Task No. 5 | Improve Results

- Improve Results: 
    - Design a test harness to select from the models with the best scores from *Task No.
        3*:
        - Customize the training metric, if needed. E.g.,
            - For an imbalanced binary classification problem, adjust the loss
                function to account for this.
        - Use hyperparameter optimization, where applicable to try and increase
            the scores from *Task No. 3*. Use Methods such as:
            - Grid search
            - Random Search
        - Use proper cross-validation methods, to be able to evaluate the model
            performance on for the model unseen data and to combat overfitting,
            if the case.
        - Try ensembles of estimators together with custom weights for each
            estimator in the ensemble.
    - Go back and forth between tasks 2-4, as needed.

## Task No. 6 | Present Results

- Finalize the model.
- Make final predictions.
- Document the process, that led to the final results.
- Present your work and explain how the final solution solves the problem given
    at the beginning.
