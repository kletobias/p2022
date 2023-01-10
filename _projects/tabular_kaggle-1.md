---
layout: distill
title: 'Deep Dive Tabular Data Pt. 1'
date: 2023-01-09
description: 'Preprocssing Data'
img: 'assets/img/838338477938@+-791693336.jpg'
tags: ['tabular data', 'fastai', 'pandas', 'tree models', 'hypterparameter optimization']
category: ['tabular data']
authors: 'Tobias Klein'
comments: true
---
<br>
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#complete-toc">Complete TOC</a></div>
    <div class="no-math"><a href="#part-1-introduction">Part 1: Introduction</a></div>
    <div class="no-math"><a href="#imports-for-the-series">Imports For The Series</a></div>
    <div class="no-math"><a href="#importing-the-flat-files">Importing The Flat Files</a></div>
    <div class="no-math"><a href="#looking-at-the-training-dataset">Looking At The Training Dataset</a></div>
    <div class="no-math"><a href="#visualizing-missing-values">Visualizing Missing Values</a></div>
    <div class="no-math"><a href="#categorical-embeddings">Categorical Embeddings</a></div>
    <div class="no-math"><a href="#final-preprocessing-tabularpandas">Final Preprocessing: TabularPandas</a></div>
  </nav>
</d-contents>

# Series: Kaggle Competition - Deep Dive Tabular Data

[**Deep Dive Tabular Data Part 1**]({% link _projects/tabular_kaggle-1.md %})<br>
[**Deep Dive Tabular Data Part 2**]({% link _projects/tabular_kaggle-2.md %})<br>
[**Deep Dive Tabular Data Part 3**]({% link _projects/tabular_kaggle-3.md %})<br>
[**Deep Dive Tabular Data Part 4**]({% link _projects/tabular_kaggle-4.md %})<br>
[**Deep Dive Tabular Data Part 5**]({% link _projects/tabular_kaggle-5.md %})<br>
[**Deep Dive Tabular Data Part 6**]({% link _projects/tabular_kaggle-6.md %})<br>
[**Deep Dive Tabular Data Part 7**]({% link _projects/tabular_kaggle-7.md %})<br>

This series documents the process of importing raw tabular data from a CSV file,
to submitting the final predictions on the Kaggle test set for the competition.

## Complete TOC

```md
Series: Kaggle Competition - Deep Dive Tabular Data

Part 1: Introduction
  Imports For The Series
  Importing The Flat Files
  Looking At The Training Dataset
    Columns 0:9 - **Observations about the data**
    Columns 10:19
    Columns 20:29
    Columns 30:39
    Columns 40:49
    Columns 50:59
    Columns 60:69
    Columns 70:80
  Visualizing Missing Values
    Split DataFrame By Columns
    Handling Missing Values
  Categorical Embeddings
    Find Candidate Columns
    Criteria
    Unique Values
    The Transformation
  Final Preprocessing: *TabularPandas*
    Explanation Of Parameter Choices
      procs

Part 2: Tree Based Models For Interpretability
  TOC Of Part 2
    Train & Validation Splits
    Sklearn DecisionTreeRegressor
    Theory
    Feature Importance Metric Deep Dive Experiment
Part 3
    RandomForestRegressor (RFR) For Interpretation
      Create Root Mean Squared Error Metric For Scoring Models
      Custom Function To Create And Fit A RFR Estimator
      RFR - Theory
      RFR - Average RMSE By Number Of Estimators
    Out-Of-Bag Error Explained
    RFR: Standard Deviation Of RMSE By Number Of Estimators
      Visualization Using Swarmplot
      Feature Importances And Selection Using RFR
      New Training Set *xs_imp* After Feature Elimination
      Interpretation Of RMSE Values
Part 4
    Dendrogram Visualization For Spearman Rank Correlations
      Conclusion
    Dendrogram Findings Applied
      Drop One Feature At A Time
      Drop Both Features Together
      Evaluate *oob_scores*
    New Train & Validation Sets Using Resulting Feature Set
      Baseline RMSE Scores
  Exploring The Impact of Individual Columns
  Partial Dependence
  Tree Interpreter
  Out-Of-Domain Problem
  Identifying *Out-Of-Domain Data*
Part 5:    Creation Of The Kaggle Submission
    Creating Estimators Optimized For Kaggle
    RandomForestRegressor Optimization
      Final RandomForestRegressor rmse Values
    tabular_learner - Deep Learning Model
      Testing Of Different Values For Parameter max_card
      Run TabularPandas Function
      Create Dataloaders Object
      Create tabular_learner estimator
    Preprocessing Of The Kaggle Test Dataset
Part 6: Optimization Routines & Final Submission
    tabular_learner Optimization
    XGBRegressor Optimization
    Three Model Ensemble
    Kaggle Submission
```

## Part 1: Introduction

The name of the Kaggle competition the data is from and the submission is
submitted to is the [*House Prices - Advanced Regression Techniques
Competition*](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
The data can be found [*here*](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

- Number of participants in the competition (2023-01-09): **4345**
- Current position on the leader board (2023-01-09): **39**
- Problem type: **Regression**
- Number of samples in the training set: **1460**
- Number of samples in the test set: **1460**
- Number of independent variables: **80**
- Number of dependent variables: **1**
- Submission scored on the natural Logarithm of the dependent variable.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/p.png" title="Overview" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Competition Name: House Prices - Advanced Regression Techniques.<br>
        Position on the leader board, as of 2023-01-09.<br>
</div>

## Imports For The Series


```python
from dtreeviz.trees import *
from fastai.tabular.all import *
from itertools import product
from numpy.core._operand_flag_tests import inplace_add
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype
from pathlib import Path
from scipy.cluster import hierarchy as hc
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold, RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeRegressor, plot_tree
from subprocess import check_output
from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall
from xgboost import XGBRegressor
import janitor
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import pickle
import scipy
import seaborn as sns
import torch
import warnings
```


```python
plt.style.use("science")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
pd.options.display.max_rows = 6  # display fewer rows for readability
pd.options.display.max_columns = 10  # display fewer columns for readability
plt.ion() # make plt interactive
seed = 42
```

## Importing The Flat Files
The train and validation dataset from kaggle is imported and the header names
are cleaned. In this case, the headers have camel case words. While this can be
good for readability, it is of no use for machine learning and is therefore
changed to all lowercase instead.


```python
base = check_output(["zsh", "-c", "echo $UKAGGLE"]).decode("utf-8")[:-1]
traind = base + "/" + "my_competitions/kaggle_competition_house_prices/data/train.csv"
testd = base + "/" + "my_competitions/kaggle_competition_house_prices/data/test.csv"
train = pd.read_csv(traind, low_memory=False).clean_names()
valid = pd.read_csv(testd, low_memory=False).clean_names()
```

Kaggle evaluates all submissions, after the dependent variable `saleprice` is
transformed by applying the Logarithm to all its values. For training, the
dependent variable is also transformed using the natural Logarithm.


```python
def tl(df):
    df["saleprice"] = df["saleprice"].apply(lambda x: np.log(x))
    return df

tl(train)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>mssubclass</th>
      <th>mszoning</th>
      <th>lotfrontage</th>
      <th>lotarea</th>
      <th>...</th>
      <th>mosold</th>
      <th>yrsold</th>
      <th>saletype</th>
      <th>salecondition</th>
      <th>saleprice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>...</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.247694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>...</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.109011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>...</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.317167</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>RL</td>
      <td>66.0</td>
      <td>9042</td>
      <td>...</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.493130</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>9717</td>
      <td>...</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>11.864462</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9937</td>
      <td>...</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>11.901583</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 81 columns</p>
</div>



## Looking At The Training Dataset

We look at a sample of the training data, in order to get a better idea of how
the data in this dataset is given. There are 81 columns in this dataset and
therefore we will need to look several slices of the set of columns in order
to analyse the data in each column.


```python
train.columns.__len__()
```




    81



### Columns 0:9 - **Observations about the data**

An example of how the data was analyzed for each column, is given for the
first 10 columns below. This analysis helps in understanding which columns are
of type categorical and which are of type continuous.
In this dataset most columns are of type categorical, with only few columns,
that are classified as being high cardinality. This is regardless of whether
they are of type object (strings and numerical characters are found in the
values.) or numerical.

- `id`: Is a standard integer range index, that likely increases monotonously by
   row and might have unique values for each building in the dataset.
- `mssubclass`: Gives the building class. It looks like a categorical column,
   with numerical classes.
- `mszoning`: Is the general zoning classification of a building and it looks
- like a categorical column with string classes.
- `lotfrontage`: Gives the street frontage, that each building has towards a
   street. This is measured by taking the horizontal distance, that is
   perpendicular to the side of the street and that measures the length that the
   side of the house facing the street shares with the street.
- `lotarea`: Gives lot size of the property in square feet.
- `street`: Gives the type of access road for the building.
- `alley`: Is the street type of the alley access associated with the property.
- `lotshape`: Is the general shape of the property.
- `landcontour`: Measures the slope of the property. Categorical column with
   string type classes.
- `utilities`: The type of the utilities available for the property. Likely a
   categorical column with string classes.

We print a sample of five rows for the first three columns in the training
dataset, in order to get a better understanding of their values.


```python
train.iloc[:, :3].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>mssubclass</th>
      <th>mszoning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>893</td>
      <td>20</td>
      <td>RL</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>1106</td>
      <td>60</td>
      <td>RL</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>30</td>
      <td>RM</td>
    </tr>
    <tr>
      <th>522</th>
      <td>523</td>
      <td>50</td>
      <td>RM</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>1037</td>
      <td>20</td>
      <td>RL</td>
    </tr>
  </tbody>
</table>
</div>



A custom function takes ten columns of a DataFrame `df` as input, defined by
the lower limit `ll`, upper limit `ul` and a cutoff, that limits the output of
the unique values to columns with less than the integer specified in parameter
`ll`.


```python
def auq(df: pd.DataFrame, ll, ul, lt):
    out = [
        (
            c,
            f"Number of unique values: {df[c].nunique()}",
            f"Number of [non-missing values, missing values]: {df[c].isna().value_counts().tolist()}",
            f"The complete list of unique values: {df[c].unique().tolist()}",
        )
        for c in df.iloc[:, ll:ul].columns
        if len(df[c].unique()) < lt  # output trimmed for readability
    ]
    dictuq = {}
    for o in out:
        dictuq[o[0]] = (o[1], o[2], o[3])
        print(f"Column (has less than {lt} unique values): {o[0]}")
        for i in range(3):
            print(f"{dictuq[o[0]][i]}")
        print()
    return dictuq
```


```python
out = auq(train, 0, 9, 8)
```

    Column (has less than 8 unique values): mszoning
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['RL', 'RM', 'C (all)', 'FV', 'RH']
    
    Column (has less than 8 unique values): street
    Number of unique values: 2
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Pave', 'Grvl']
    
    Column (has less than 8 unique values): alley
    Number of unique values: 2
    Number of [non-missing values, missing values]: [1369, 91]
    The complete list of unique values: [nan, 'Grvl', 'Pave']
    
    Column (has less than 8 unique values): lotshape
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Reg', 'IR1', 'IR2', 'IR3']
    
    Column (has less than 8 unique values): landcontour
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Lvl', 'Bnk', 'Low', 'HLS']
    


### Columns 10:19


```python
train.iloc[:, 10:20].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lotconfig</th>
      <th>landslope</th>
      <th>neighborhood</th>
      <th>condition1</th>
      <th>condition2</th>
      <th>bldgtype</th>
      <th>housestyle</th>
      <th>overallqual</th>
      <th>overallcond</th>
      <th>yearbuilt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Sawyer</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1963</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>Corner</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>1994</td>
    </tr>
    <tr>
      <th>413</th>
      <td>Inside</td>
      <td>Gtl</td>
      <td>OldTown</td>
      <td>Artery</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1927</td>
    </tr>
    <tr>
      <th>522</th>
      <td>Corner</td>
      <td>Gtl</td>
      <td>BrkSide</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>6</td>
      <td>7</td>
      <td>1947</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Timber</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>9</td>
      <td>5</td>
      <td>2007</td>
    </tr>
  </tbody>
</table>
</div>




```python
out.update(auq(train, 10, 20, 8))
```

    Column (has less than 8 unique values): lotconfig
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3']
    
    Column (has less than 8 unique values): landslope
    Number of unique values: 3
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Gtl', 'Mod', 'Sev']
    
    Column (has less than 8 unique values): bldgtype
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs']
    


### Columns 20:29


```python
train.iloc[:, 20:30].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yearremodadd</th>
      <th>roofstyle</th>
      <th>roofmatl</th>
      <th>exterior1st</th>
      <th>exterior2nd</th>
      <th>masvnrtype</th>
      <th>masvnrarea</th>
      <th>exterqual</th>
      <th>extercond</th>
      <th>foundation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>2003</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>1995</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>BrkFace</td>
      <td>362.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>WdShing</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
    </tr>
    <tr>
      <th>522</th>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>CBlock</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>2008</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>Stone</td>
      <td>70.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
    </tr>
  </tbody>
</table>
</div>




```python
out.update(auq(train, 20, 30, 8))
```

    Column (has less than 8 unique values): roofstyle
    Number of unique values: 6
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed']
    
    Column (has less than 8 unique values): masvnrtype
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1452, 8]
    The complete list of unique values: ['BrkFace', 'None', 'Stone', 'BrkCmn', nan]
    
    Column (has less than 8 unique values): exterqual
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Gd', 'TA', 'Ex', 'Fa']
    
    Column (has less than 8 unique values): extercond
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['TA', 'Gd', 'Fa', 'Po', 'Ex']
    
    Column (has less than 8 unique values): foundation
    Number of unique values: 6
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone']
    


### Columns 30:39


```python
train.iloc[:, 30:40].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bsmtqual</th>
      <th>bsmtcond</th>
      <th>bsmtexposure</th>
      <th>bsmtfintype1</th>
      <th>bsmtfinsf1</th>
      <th>bsmtfintype2</th>
      <th>bsmtfinsf2</th>
      <th>bsmtunfsf</th>
      <th>totalbsmtsf</th>
      <th>heating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>663</td>
      <td>Unf</td>
      <td>0</td>
      <td>396</td>
      <td>1059</td>
      <td>GasA</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>Ex</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>1032</td>
      <td>Unf</td>
      <td>0</td>
      <td>431</td>
      <td>1463</td>
      <td>GasA</td>
    </tr>
    <tr>
      <th>413</th>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>1008</td>
      <td>1008</td>
      <td>GasA</td>
    </tr>
    <tr>
      <th>522</th>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>399</td>
      <td>Unf</td>
      <td>0</td>
      <td>605</td>
      <td>1004</td>
      <td>GasA</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>Ex</td>
      <td>TA</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>1022</td>
      <td>Unf</td>
      <td>0</td>
      <td>598</td>
      <td>1620</td>
      <td>GasA</td>
    </tr>
  </tbody>
</table>
</div>




```python
out.update(auq(train, 30, 40, 8))
```

    Column (has less than 8 unique values): bsmtqual
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1423, 37]
    The complete list of unique values: ['Gd', 'TA', 'Ex', nan, 'Fa']
    
    Column (has less than 8 unique values): bsmtcond
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1423, 37]
    The complete list of unique values: ['TA', 'Gd', nan, 'Fa', 'Po']
    
    Column (has less than 8 unique values): bsmtexposure
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1422, 38]
    The complete list of unique values: ['No', 'Gd', 'Mn', 'Av', nan]
    
    Column (has less than 8 unique values): bsmtfintype1
    Number of unique values: 6
    Number of [non-missing values, missing values]: [1423, 37]
    The complete list of unique values: ['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', nan, 'LwQ']
    
    Column (has less than 8 unique values): bsmtfintype2
    Number of unique values: 6
    Number of [non-missing values, missing values]: [1422, 38]
    The complete list of unique values: ['Unf', 'BLQ', nan, 'ALQ', 'Rec', 'LwQ', 'GLQ']
    
    Column (has less than 8 unique values): heating
    Number of unique values: 6
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']
    


### Columns 40:49


```python
train.iloc[:, 40:50].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>heatingqc</th>
      <th>centralair</th>
      <th>electrical</th>
      <th>1stflrsf</th>
      <th>2ndflrsf</th>
      <th>lowqualfinsf</th>
      <th>grlivarea</th>
      <th>bsmtfullbath</th>
      <th>bsmthalfbath</th>
      <th>fullbath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1068</td>
      <td>0</td>
      <td>0</td>
      <td>1068</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1500</td>
      <td>1122</td>
      <td>0</td>
      <td>2622</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>413</th>
      <td>Gd</td>
      <td>Y</td>
      <td>FuseA</td>
      <td>1028</td>
      <td>0</td>
      <td>0</td>
      <td>1028</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>522</th>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1004</td>
      <td>660</td>
      <td>0</td>
      <td>1664</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1620</td>
      <td>0</td>
      <td>0</td>
      <td>1620</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Columns 50:59


```python
out.update(auq(train, 40, 50, 8))
```

    Column (has less than 8 unique values): heatingqc
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Ex', 'Gd', 'TA', 'Fa', 'Po']
    
    Column (has less than 8 unique values): centralair
    Number of unique values: 2
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Y', 'N']
    
    Column (has less than 8 unique values): electrical
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1459, 1]
    The complete list of unique values: ['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix', nan]
    
    Column (has less than 8 unique values): bsmtfullbath
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: [1, 0, 2, 3]
    
    Column (has less than 8 unique values): bsmthalfbath
    Number of unique values: 3
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: [0, 1, 2]
    
    Column (has less than 8 unique values): fullbath
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: [2, 1, 3, 0]
    



```python
train.iloc[:, 50:60].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>halfbath</th>
      <th>bedroomabvgr</th>
      <th>kitchenabvgr</th>
      <th>kitchenqual</th>
      <th>totrmsabvgrd</th>
      <th>functional</th>
      <th>fireplaces</th>
      <th>fireplacequ</th>
      <th>garagetype</th>
      <th>garageyrblt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1963.0</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>2</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1994.0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1927.0</td>
    </tr>
    <tr>
      <th>522</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>2</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1950.0</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Ex</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Ex</td>
      <td>Attchd</td>
      <td>2008.0</td>
    </tr>
  </tbody>
</table>
</div>



### Columns 60:69


```python
train.iloc[:, 60:70].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>garagefinish</th>
      <th>garagecars</th>
      <th>garagearea</th>
      <th>garagequal</th>
      <th>garagecond</th>
      <th>paveddrive</th>
      <th>wooddecksf</th>
      <th>openporchsf</th>
      <th>enclosedporch</th>
      <th>3ssnporch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>RFn</td>
      <td>1</td>
      <td>264</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>RFn</td>
      <td>2</td>
      <td>712</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>186</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>Unf</td>
      <td>2</td>
      <td>360</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>130</td>
      <td>0</td>
    </tr>
    <tr>
      <th>522</th>
      <td>Unf</td>
      <td>2</td>
      <td>420</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>24</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>Fin</td>
      <td>3</td>
      <td>912</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>228</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
out.update(auq(train, 60, 70, 8))
```

    Column (has less than 8 unique values): garagefinish
    Number of unique values: 3
    Number of [non-missing values, missing values]: [1379, 81]
    The complete list of unique values: ['RFn', 'Unf', 'Fin', nan]
    
    Column (has less than 8 unique values): garagecars
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: [2, 3, 1, 0, 4]
    
    Column (has less than 8 unique values): garagequal
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1379, 81]
    The complete list of unique values: ['TA', 'Fa', 'Gd', nan, 'Ex', 'Po']
    
    Column (has less than 8 unique values): garagecond
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1379, 81]
    The complete list of unique values: ['TA', 'Fa', nan, 'Gd', 'Po', 'Ex']
    
    Column (has less than 8 unique values): paveddrive
    Number of unique values: 3
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Y', 'N', 'P']
    


### Columns 70:80


```python
train.iloc[:, 70:81].sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screenporch</th>
      <th>poolarea</th>
      <th>poolqc</th>
      <th>fence</th>
      <th>miscfeature</th>
      <th>...</th>
      <th>mosold</th>
      <th>yrsold</th>
      <th>saletype</th>
      <th>salecondition</th>
      <th>saleprice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>...</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>11.947949</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.691580</td>
    </tr>
    <tr>
      <th>413</th>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>11.652687</td>
    </tr>
    <tr>
      <th>522</th>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>10</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>11.976659</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>9</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.661914</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11 columns</p>
</div>




```python
out.update(auq(train, 70, 80, 8))
```

    Column (has less than 8 unique values): poolqc
    Number of unique values: 3
    Number of [non-missing values, missing values]: [1453, 7]
    The complete list of unique values: [nan, 'Ex', 'Fa', 'Gd']
    
    Column (has less than 8 unique values): fence
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1179, 281]
    The complete list of unique values: [nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw']
    
    Column (has less than 8 unique values): miscfeature
    Number of unique values: 4
    Number of [non-missing values, missing values]: [1406, 54]
    The complete list of unique values: [nan, 'Shed', 'Gar2', 'Othr', 'TenC']
    
    Column (has less than 8 unique values): yrsold
    Number of unique values: 5
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: [2008, 2007, 2006, 2009, 2010]
    
    Column (has less than 8 unique values): salecondition
    Number of unique values: 6
    Number of [non-missing values, missing values]: [1460]
    The complete list of unique values: ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']
    


## Visualizing Missing Values

The `missingno` library offers a powerful visualization tool, that helps us
understand better which columns have missing values, how many and in which rows
they are found.

### Split DataFrame By Columns

The function itself works best for data, with less than or equal to fifty
columns per plot. Otherwise, even with rotated labels, labels get overlapped by
other labels and thus can not be identified anymore.<br>
The custom function `fp_msno` splits a DataFrame with less than 100 columns in
total into two parts and returns the rendered plots for both halves separately.


```python
def fp_msno(df: pd.DataFrame, num=1):
    '''Visualize missing values in the DataFrame'''
    len_cols = len(df.columns)
    the_dict = dict(
        start=0, first_half_uplimt=[], second_half_llimt=[], second_half_uplimt=[]
    )
    split_at = np.floor_divide(len_cols, 2)
    the_dict["first_half_uplimt"].append(split_at)
    the_dict["second_half_llimt"].append(split_at + 1)
    the_dict["second_half_uplimt"].append(len_cols)
    print(the_dict["second_half_uplimt"])
    print(int(len_cols))
    print(the_dict["first_half_uplimt"][0])
    assert the_dict["second_half_uplimt"][0] == int(len_cols)
    msno.matrix(
        df.loc[
            :, [col for col in df.columns if df.columns.tolist().index(col) < split_at]
        ],
        figsize=(11,6),
        label_rotation=60
    )
    plt.xticks(fontsize=12)
    plt.subplots_adjust(top=.8)
    plt.savefig(f"{num}", dpi=300)
    plt.show()
    plt.savefig(f"missingno-matrix{num}", dpi=300)
    msno.matrix(
        df.loc[
            :,
            [
                col
                for col in df.columns
                if split_at <= df.columns.tolist().index(col) <= len_cols
            ],
        ],
        figsize=(11,6),
        label_rotation=60
    )
    plt.xticks(fontsize=12)
    plt.subplots_adjust(top=.8)
    plt.savefig("2", dpi=300)
    plt.show()
```

### Handling Missing Values

We call the function and pass it the training dataset `train`.


```python
fp_msno(train)
```

    [81]
    81
    40



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_39_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Missingno 1/2
</div>
    






    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_39_3.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Missingno 2/2
</div>
    


#### Overview

The output gives a visual overview of how many missing values each column has.
It also shows in which rows the missing values occur and therefore lets one
get an idea of which columns are correlated in terms of their rows with
missing values.

Using the plot, we can investigate the following, in order to see if there is
a pattern present for each column, that has many missing values. While 'many
missing values' is by no means a criterium to go by when filling missing
values, there is no *fits all* type of solution when it comes to dealing with
missing values.

#### Default Strategy

Let me explain why and how I determine which columns to manually fill, using
*custom* fill values, that are not the median of all non-missing values found in
a column, if the column is of type *numerical* or the mode in case of a string
column.

#### Early Benchmark

I like to use as little preprocessing as possible before training the first
model, just enough that the data is ready for the model to be trained on. That
way I get a first benchmark early. Most of the time I use a tree based model,
that lets me evaluate the relative feature importances of the columns among
other metrics, that I look at.
Only after this first iteration, will I think about if using custom fill values
is likely to payoff in bettering the accuracy score, more than all the other
preprocessing steps that could be added instead.

#### Examples Of Common Cases

For this article, in order to show how I would deal with the columns that have
missing values, we look at the ones, where rows with missing values actually
don't mean, that the value is missing - is unknown. This happens frequently,
when the data is not properly collected for that feature with missing values.
E.g., column `poolqc`, which stands for 'pool quality'.

We create the function `bivar`, that accepts two columns of a DataFrame,
`related_col` and `missing_col` and returns the unique values for the pair of
the two columns, only using the rows where `missing_col` has missing values.


```python
def bivar(related_col, missing_col):
    missing_rows = train[[related_col, missing_col]][train[missing_col].isna() == True]
    uniques = {related_col: [], missing_col: []}
    for key in uniques.keys():
        uniques[key].append(missing_rows[key].unique())
    print(uniques)
```

##### Case: A Category Is Missing

The output for `poolarea` and `poolqc` shows, that only rows with missing values
in `poolqc` are found, if and only if column `poolarea` has value 0 in the same
row. A value of 0 means, that the listing has no pool and therefore no `poolqc`.
The values are not missing in this case, instead a new class should be added to
the ones present in the column. It could be something like 'no_pool'.


```python
bivar("poolarea", "poolqc")
```

    {'poolarea': [array([0])], 'poolqc': [array([nan], dtype=object)]}


The same relationship is found for column `fireplacequ`, which categorizes the
quality of a fireplace if the particular house has a fireplace.


```python
bivar("fireplaces", "fireplacequ")
```

    {'fireplaces': [array([0])], 'fireplacequ': [array([nan], dtype=object)]}


##### Case: Default Fill Values

Column `alley` uses `nan` values where neither a gravel nor a pave alley is
found. A new category, like 'no_alley' could be used to fill the missing values
here.


```python
train.alley.unique()
```




    array([nan, 'Grvl', 'Pave'], dtype=object)



Variable `lotfrontage` measures the *Lot Frontage*, which is a legal term.
Without getting too specific, it measures the portion of a lot abutting a
street. It is unclear, whether the `nan` values in this column are not just
actual `nan` values. In this case I would prefer to use the median of all
non-missing values in the column, to fill the missing values.


```python
train.lotfrontage.unique()
```




    array([ 65.,  80.,  68.,  60.,  84.,  85.,  75.,  nan,  51.,  50.,  70.,
            91.,  72.,  66., 101.,  57.,  44., 110.,  98.,  47., 108., 112.,
            74., 115.,  61.,  48.,  33.,  52., 100.,  24.,  89.,  63.,  76.,
            81.,  95.,  69.,  21.,  32.,  78., 121., 122.,  40., 105.,  73.,
            77.,  64.,  94.,  34.,  90.,  55.,  88.,  82.,  71., 120., 107.,
            92., 134.,  62.,  86., 141.,  97.,  54.,  41.,  79., 174.,  99.,
            67.,  83.,  43., 103.,  93.,  30., 129., 140.,  35.,  37., 118.,
            87., 116., 150., 111.,  49.,  96.,  59.,  36.,  56., 102.,  58.,
            38., 109., 130.,  53., 137.,  45., 106., 104.,  42.,  39., 144.,
           114., 128., 149., 313., 168., 182., 138., 160., 152., 124., 153.,
            46.])



For variable `fence`, I would use the mode, since there is no indication, that
the `nan` values are not just missing values.


```python
train.fence.unique()
```




    array([nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], dtype=object)



The examples cover many of the cases, that can happen and show for each of the
most common types how I would fill the missing values. A more in-depth analysis
and a custom fill strategy can be necessary, if the particular column is of high
importance and therefore justifies a complex filling strategy.

## Categorical Embeddings

The goal is to create categorical embeddings, that will transform the
categorical string variables into ordered categorical type columns. For this, we
are looking for columns whose unique values have an inherent order, that we can
use to make the transformation using the `pandas` library. While categorical
embeddings don't make a difference for tree ensembles, they do for deep learning
models, that unlike tree based models can make use of the added information
given by categorical embeddings for categorical variables.

### Find Candidate Columns
The output below shows, that there are three types of data types currently
assigned to the columns of `train`. The ones with dtype `object` are the ones we
focus on here in the following step.


```python
def guc(train: pd.DataFrame, p=None):
    dts = [train[c].dtype for c in train.columns]
    ut, uc = np.unique(dts, return_counts=True)
    if p == True:
        for i in range(len(ut)):
            print(ut[i], uc[i])

    return ut, uc

_, _ = guc(train, p=True)
```

    int64 34
    float64 4
    object 43


The function below returns a dictionary of column name and list of unique values
for all columns, that have less than `ul` unique values and that are of type
'object' by default. The input parameters can be altered to filter for columns
with less or more number of unique values than the default value and one can
look for other column data types as well.


```python
def can(train=train, ul=13, tp="object"):
    dd = [
        (c, train[c].unique())
        for c in train.columns
        if train[c].nunique() < ul and train[c].dtype == tp
    ]
    ddc = dict(dd)
    return ddc

ddc = can()
ddc
```




    {'mszoning': array(['RL', 'RM', 'C (all)', 'FV', 'RH'], dtype=object),
     'street': array(['Pave', 'Grvl'], dtype=object),
     'alley': array([nan, 'Grvl', 'Pave'], dtype=object),
     'lotshape': array(['Reg', 'IR1', 'IR2', 'IR3'], dtype=object),
     'landcontour': array(['Lvl', 'Bnk', 'Low', 'HLS'], dtype=object),
     'utilities': array(['AllPub', 'NoSeWa'], dtype=object),
     'lotconfig': array(['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'], dtype=object),
     'landslope': array(['Gtl', 'Mod', 'Sev'], dtype=object),
     'condition1': array(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA',
            'RRNe'], dtype=object),
     'condition2': array(['Norm', 'Artery', 'RRNn', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe'],
           dtype=object),
     'bldgtype': array(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], dtype=object),
     'housestyle': array(['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf',
            '2.5Fin'], dtype=object),
     'roofstyle': array(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'], dtype=object),
     'roofmatl': array(['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv',
            'Roll', 'ClyTile'], dtype=object),
     'masvnrtype': array(['BrkFace', 'None', 'Stone', 'BrkCmn', nan], dtype=object),
     'exterqual': array(['Gd', 'TA', 'Ex', 'Fa'], dtype=object),
     'extercond': array(['TA', 'Gd', 'Fa', 'Po', 'Ex'], dtype=object),
     'foundation': array(['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'], dtype=object),
     'bsmtqual': array(['Gd', 'TA', 'Ex', nan, 'Fa'], dtype=object),
     'bsmtcond': array(['TA', 'Gd', nan, 'Fa', 'Po'], dtype=object),
     'bsmtexposure': array(['No', 'Gd', 'Mn', 'Av', nan], dtype=object),
     'bsmtfintype1': array(['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', nan, 'LwQ'], dtype=object),
     'bsmtfintype2': array(['Unf', 'BLQ', nan, 'ALQ', 'Rec', 'LwQ', 'GLQ'], dtype=object),
     'heating': array(['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], dtype=object),
     'heatingqc': array(['Ex', 'Gd', 'TA', 'Fa', 'Po'], dtype=object),
     'centralair': array(['Y', 'N'], dtype=object),
     'electrical': array(['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix', nan], dtype=object),
     'kitchenqual': array(['Gd', 'TA', 'Ex', 'Fa'], dtype=object),
     'functional': array(['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'], dtype=object),
     'fireplacequ': array([nan, 'TA', 'Gd', 'Fa', 'Ex', 'Po'], dtype=object),
     'garagetype': array(['Attchd', 'Detchd', 'BuiltIn', 'CarPort', nan, 'Basment', '2Types'],
           dtype=object),
     'garagefinish': array(['RFn', 'Unf', 'Fin', nan], dtype=object),
     'garagequal': array(['TA', 'Fa', 'Gd', nan, 'Ex', 'Po'], dtype=object),
     'garagecond': array(['TA', 'Fa', nan, 'Gd', 'Po', 'Ex'], dtype=object),
     'paveddrive': array(['Y', 'N', 'P'], dtype=object),
     'poolqc': array([nan, 'Ex', 'Fa', 'Gd'], dtype=object),
     'fence': array([nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], dtype=object),
     'miscfeature': array([nan, 'Shed', 'Gar2', 'Othr', 'TenC'], dtype=object),
     'saletype': array(['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth'],
           dtype=object),
     'salecondition': array(['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'],
           dtype=object)}



The output of the function shows, that there are many columns, that could be
good candidates for categorical embeddings. We look for ones, that have a clear
inherent ordering and that can be batch processed.

### Criteria
Using the output of the function above, the unique values of all columns of type
`object` are screened for ones that have an intrinsic ordering. One could spend
more time finding additional columns with ordered values of type `object` or
also including columns with unique values of other types, e.g., see the output
of the `can` function above. The goal here was to find several columns that meet
this criteria, that can be batch processed without the need to individually set
the ordered categorical dtype for each column, since the unique values of the
columns don't share close to the same set of unique values, that all follow the
same order.

### Unique Values
The unique values below are found in several string columns, follow the same
ordering and are almost identical. The abbreviations are listed in the table
below. The placement within the order of each set of unique values for each
column depends on whether there are missing values present in the value range of
the particular column. This is indicated by the values of column *Condition For
Rank* in the table below. Only the relative rank of each class within the values
found in a specific column is relevant, e.g., whether the order of *"Po"* is the
lowest (0) or (1), second lowest, does not matter for the actual transformation.

| Order  | Abbreviation  | Category                 | Condition For Rank |
| :----: | :-----------: | ----------               | :---------:        |
| 0      | "FM"          | false or missing (`nan`) | -                  |
| 0      | None          | missing value (`nan`)    | -                  |
| 0/1    | "Po"          | poor                     | -+ `nan`           |
| 1/2    | "Fa"          | fair                     | -+ `nan`           |
| 2/3    | "TA"          | typical/average          | -+ `nan`           |
| 3/4    | "Gd"          | good                     | -+ `nan`           |
| 4/5    | "Ex"          | excellent                | -+ `nan`           |

### The Transformation
Three different variations of the value range are found and specified below in
ascending order from left to right.


```python
uset = ["Po", "Fa", "TA", "Gd", "Ex"]
usetNA = [None, "Po", "Fa", "TA", "Gd", "Ex"]
usetna = ["FM", "Po", "Fa", "TA", "Gd", "Ex"]
```

The function `cu` does the transformation for all three cases.


```python
def cu(df: pd.DataFrame, uset: list, usetna: list):
    cols = [
        (c, df[c].unique().tolist())
        for c in df.columns.tolist()
        if df[c].dtype == "object" and set(df[c].unique()).issuperset(uset)
    ]
    for cc in cols:
        # name of the column, as filtered by list comprehension cols
        name = cc[0]
        vals = uset
        valsna = usetna
        # fill missing values with string "FM" - false or missing
        df[name].fillna(value="FM", inplace=True)
        # change column to dtype category
        df[name] = df[name].astype("category")
        # no missing values and no FM category indicating missing values.
        if set(df[name].unique()) == set(uset):
            # unique values are not changed in this case.
            vals = vals
            # dtype changed to ordered categorical.
            df[name].cat.set_categories(vals, ordered=True, inplace=True)
            # print name of column, followed by its ordered categories.
            print(name, df[name].cat.categories)
        # missing values present, as indicated by category "FM"
        elif set(df[name].unique()) == set(usetna):
            valsna = valsna
            df[name].cat.set_categories(valsna, ordered=True, inplace=True)
            # print name of column, followed by its ordered categories.
            print(name, df[name].cat.categories)
    return df
```

We pass the function the training dataset and the three value ranges to filter
on and update the `train` DataFrame with the output of function `cu`.


```python
train = cu(train, uset, usetna)
```

    extercond Index(['Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    heatingqc Index(['Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    fireplacequ Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    garagequal Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    garagecond Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')


## Final Preprocessing: *TabularPandas*

Often times I tend to use function `train_test_split` from sklearn, after
manually doing all the preprocessing. In this case, we use function
`TabularPandas` ([*TabularPandas
Documentation*](https://docs.fast.ai/tabular.core.html#tabularpandas)) from
library `fastai`, which is a convenient wrapper around several preprocessing
functions, that transform a DataFrame.


```python
procs = [Categorify, FillMissing]
dep_var = "saleprice"
cont, cat = cont_cat_split(df=train, max_card=1, dep_var=dep_var)
train_s, valid_s = RandomSplitter(seed=42)(train)
to = TabularPandas(
    train,
    procs,
    cat,
    cont,
    y_names=dep_var,
    splits=(train_s, valid_s),
    reduce_memory=False,
)
```

### Explanation Of Parameter Choices

For the `train` DataFrame we specify the following parameters and add a
description of what each one does.

#### procs
We specify `[Categorify, FillMissing]` for this parameter. What that does is
apply two transformations to the entire input DataFrame. `Categorify` will
transform categorical variables to numerical values and will respect ones that
are already of type ordered categorical. The documentation can be found here:
[*Categorify Documentation*](https://docs.fast.ai/tabular.core.html#categorify).
`FillMissing` will only target numerical columns and use the *median* as fill
value by default. One can find its documentation following this link:
[*FillMissing Documentation*](https://docs.fast.ai/tabular.core.html#fillmissing).
Another transformation it does is add a new column for every column with missing
values, that indicates for each row in the dataset whether the value in that
column was filled or not. It does so by adding a boolean type column for every
column with missing values. E.g., see the output of the command below.


```python
xs, y = to.train.xs, to.train.y  # defined here to show output of next line.
xs.filter(like="_na", axis=1).sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lotfrontage_na</th>
      <th>masvnrarea_na</th>
      <th>garageyrblt_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>465</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>574</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Parameter `dep_var` tells `TabularPandas`, which column(s) to use as the
dependent variable(s).

`cont`, `cat` are the outputs of function `cont_cat_split` ([*cont_cat_split
Documentation*](https://docs.fast.ai/tabular.core.html#cont_cat_split)), which
will assign each column to either type *continuous* or type *categorical* based
on the threshold (`max_card`) passed for the maximum number of unique values
that a column may have to still be classified as being categorical. In this
case, since the preprocessing is only used to train tree ensemble models, there
is no added information for the model, if we tell it that there are continuous,
as well as categorical variables. This is will change later, when we train a
neural network, there the distinction is relevant.

`train_s`, `valid_s` are assigned the output of function `RandomSplitter` which
splits the dataset along the row index into training and test data (`valid_s`).
The default is 80 percent of rows are used for the training set, with the
remaining 20 percent used for the test set. One can specify the argument `seed`
in order to get the same split across several executions of the code.

In order to see the transformations done by calling `TabularPandas`, one has to
use the variable assigned to the output of `TabularPandas`, `to` in this case.
Like that, one can access several attributes of the created object, such as the
following.

Get the length of the training and test set.


```python
len(to.train), len(to.valid)
```




    (1168, 292)



Print the classes behind the now numerical unique values of categorical type
columns.


```python
to.classes["heatingqc"]
```




    ['#na#', 'Po', 'Fa', 'TA', 'Gd', 'Ex']




```python
to.classes["saletype"]
```




    ['#na#', 'COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth', 'WD']




```python
to["saletype"].unique()
```




    array([9, 7, 1, 2, 8, 4, 6, 5, 3], dtype=int8)




```python
to.items.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>mssubclass</th>
      <th>mszoning</th>
      <th>lotfrontage</th>
      <th>lotarea</th>
      <th>...</th>
      <th>salecondition</th>
      <th>saleprice</th>
      <th>lotfrontage_na</th>
      <th>masvnrarea_na</th>
      <th>garageyrblt_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1172</th>
      <td>1173</td>
      <td>160</td>
      <td>2</td>
      <td>35.0</td>
      <td>4017</td>
      <td>...</td>
      <td>5</td>
      <td>12.054668</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>1314</td>
      <td>60</td>
      <td>4</td>
      <td>108.0</td>
      <td>14774</td>
      <td>...</td>
      <td>5</td>
      <td>12.716402</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1327</th>
      <td>1328</td>
      <td>20</td>
      <td>4</td>
      <td>60.0</td>
      <td>6600</td>
      <td>...</td>
      <td>5</td>
      <td>11.779129</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>30</td>
      <td>5</td>
      <td>60.0</td>
      <td>6324</td>
      <td>...</td>
      <td>5</td>
      <td>11.134589</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>398</td>
      <td>60</td>
      <td>4</td>
      <td>69.0</td>
      <td>7590</td>
      <td>...</td>
      <td>5</td>
      <td>12.040608</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>



Once the preprocessing is done, one can save the resulting object to a *pickle*
file like shown below.


```python
save_pickle("top.pkl", to)
```

<!-- --- -->
<!-- layout: distill -->
<!-- title: 'Deep Dive Tabular Data Pt. 1' -->
<!-- date: 2023-01-09 -->
<!-- description: 'Preprocssing Data' -->
<!-- img: 'assets/img/838338477938@+-791693336.jpg' -->
<!-- tags: ['tabular data', 'fastai', 'pandas', 'tabular data', 'hypterparameter optimization'] -->
<!-- category: ['tabular data'] -->
<!-- authors: 'Tobias Klein' -->
<!-- comments: true -->
<!-- --- -->
<!-- <br> -->


# Part 2: Tree Based Models For Interpretability

With the preprocessing done, we can start with the machine learning. However,
this is not the type of machine learning where we look for the highest accuracy
we can get, given a model architecture and hyperparameters. In this case we are
looking to get a better understanding of which of the 80 independent variables
contribute most to the final predictions that the fitted model makes and how.

## TOC Of Part 2

```md
  Tree Based Models For Interpretability
    Train & Validation Splits
    Sklearn DecisionTreeRegressor
    Theory
    Feature Importance Metric Deep Dive Experiment
    RandomForestRegressor (RFR) For Interpretation
    Out-Of-Bag Error Explained
    RFR: Standard Deviation Of RMSE By Number Of Estimators
    Dendrogram Visualization For Spearman Rank Correlations
    Dendrogram Findings Applied
    New Train & Validation Sets Using Resulting Feature Set
  Exploring The Impact of Individual Columns
  Partial Dependence
  Tree Interpreter
    Identifying *Out-Of-Domain Data*
  Creation Of The Kaggle Submission
    Creating Estimators Optimized For Kaggle
    RandomForestRegressor Optimization
    tabular_learner - Deep Learning Model
    Preprocessing Of The Kaggle Test Dataset
    tabular_learner Optimization
    XGBRegressor Optimization
    Three Model Ensemble
    Kaggle Submission
```

### Train & Validation Splits

We will always use `to.train.xs` and `to.train.y` to assign the training pair of
independent variables and dependent variable. In the same way, we will always
assign the test pair.


```python
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y
```

### Sklearn DecisionTreeRegressor

The first model used is a simple decision tree model, that we will use to see
what columns and values the model uses to create the splits. Plus, we like to
see the order of the splits that the model chooses. From library `scikit-learn`
we use the `DecisionTreeRegressor` model ([*DecisionTreeRegressor
Documentation*](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn-tree-decisiontreeregressor)).

#### Feature Selection

Since we are looking for eliminate irrelevant variables (features), we tell it
to only consider at maximum 40 of the 80 columns when deciding which feature is
the best one to use for the next split. The model must not create leaves, that
have less than 50 samples for the final split. The model is fit using the train
split we created earlier.


```python
m = DecisionTreeRegressor(max_features=40, min_samples_leaf=50, random_state=seed)
m.fit(xs, y)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeRegressor(max_features=40, min_samples_leaf=50, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeRegressor</label><div class="sk-toggleable__content"><pre>DecisionTreeRegressor(max_features=40, min_samples_leaf=50, random_state=42)</pre></div></div></div></div></div>



#### DecisionTreeRegressor Visualization Examples

Two visualization tools are used to visualize the splits of the fitted model.
The first one is a rather basic one, while `dtreeviz` is a more detailed
visualization.


```python
ax, fig = plt.subplots(1, 1, figsize=(18, 12))
ax = plt.subplot(111)
plot_tree(m, feature_names=[c for c in xs.columns if c != "saleprice"],
          ax=ax,fontsize=6)
plt.savefig(f"plot_tree-1", dpi=300,bbox_inches='tight')
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_80_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        sklearn plot_tree
</div>
    



```python
fig = plt.figure(figsize=(22, 14))

dtreeviz(
    m,
    xs,
    y,
    xs.columns,
    dep_var,
    fontname="DejaVu Sans",
    scale=1.1,
    label_fontsize=11,
    orientation="LR",
)
```




    

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_81_0.svg" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        dtreeviz visualization
</div>
    





### Theory

A reason why tree based models are one of the best model types when it comes to
understanding the model and the splits a fitted model created lies in the fact,
that there is a good infrastructure for this type of model when it comes to
libraries and functions created for that purpose. That and the fact, that this
type of model has a transparent and fairly easy to understand structure. The
tree based models we use for understanding are ones, where the tree or the trees
in the case of the `RandomForestRegressor` are intentionally kept weak. This is
not necessarily the case for the `DecisionTreeRegressor` model, since without
limiting the number of splits, there is no mechanism to limit the size of the
final tree. While the size of the `DecisionTreeRegressor` can be limited by use
of its parameters to combat overfitting, it is the `RandomForestRegressor`
model, that is the preferred model from the two when it comes to the
`feature_importances_` metric. A metric used for *feature selection* and for
understanding the relative importance that each feature has. Both models feature
this method.

#### Feature Importances Relative Metric

From the *scikit-learn* website, one gets the following definition for the
`feature_importances_` attribute:

> property feature_importances_<br>
> Return the feature importances.
> The importance of a feature is computed as the (normalized) total
> reduction of the criterion brought by that feature. It is also known as
> the Gini importance.
>
> [*Definition of feature_importances_ attribute on scikit-learn*](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.feature_importances_)

<!-- TODO: test citation below -->
This paper<d-cite key="menze_comparison_2009"></d-cite> can be read for more on
the Gini importance metric.


```python
fi = dict(zip(train.columns.tolist(), m.feature_importances_))
```

##### Feature Selection Using Features Importance Scores

Using the feature importances values, we can set a lower limit for the feature
importance score. All features with a feature importance lower than the
threshold are dropped from the `train` dataset and saved as a new DataFrame for
illustration purposes. Only the ones are kept, that a features importance score
larger zero. The result is a subset of 11 columns of the original 80 columns.


```python
fip = pd.DataFrame(fi, index=range(0, 1))
fipl = fip[fip > 0].dropna(axis=1).T.sort_values(by=0, ascending=False)
fipl.rename(columns={0: "feature_importance"}, inplace=True)
fipl
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>kitchenabvgr</th>
      <td>0.498481</td>
    </tr>
    <tr>
      <th>garagearea</th>
      <td>0.168625</td>
    </tr>
    <tr>
      <th>halfbath</th>
      <td>0.121552</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>bedroomabvgr</th>
      <td>0.007406</td>
    </tr>
    <tr>
      <th>fireplacequ</th>
      <td>0.006791</td>
    </tr>
    <tr>
      <th>poolqc</th>
      <td>0.002268</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 1 columns</p>
</div>



### Feature Importance Metric Deep Dive Experiment

It was unclear, whether the order from most important feature to the ones with a
feature importance score of zero according to the `feature_importances_` method
changes between executions, using the same parameters and data to train the
`DecisionTreeRegressor` estimator. Changes in the subset of columns, that have a
feature importance score larger zero led to inconsistencies and ultimately
errors when the same code was executed repeatedly. This was observed while using
the same `random_state` seed to ensure, that the output of functions generating
and using pseudo random numbers stayed the same across different executions.

Another possibility is, that the feature importance scores change and ultimately
reorder the ranking of the columns based on these scores in some cases. These
deviations might however be the result of changes in the model parameters by the
user between executions, rather than returning differing values across multiple
executions without any interactions by the user.

To answer this question, an experiment is conducted where 1000
`DecisionTreeRegressor` estimators are trained and the `feature_importances_`
method is used to get and log the feature importance score for each feature
during each of the 1000 samples. The results are then averaged and the standard
deviation is calculated for each feature over the 2000 samples.

With the results, one can answer the research question of whether the
`feature_importances_` method itself is prone to generating inconsistent feature
importance scores for the features used as independent variables during fitting
of the `DecisionTreeRegressor` in this case. Using 1000 iterations will not only
give a mean feature importance score $\hat{\mu}$ for each feature that is much
closer to the true mean $\mu$, but also a value for the standard deviation
$\hat{\sigma}$ for the sample. The following function does just that and creates
two summary plots, that visualize the final results and answers the research
question.

In summary, the output shows that in this instance, there is no change in the
order of the features as given by the `feature_importances_` method. Further,
there is no relevant deviation of the values assigned to each feature by the
method, over the 1000 samples. The changes in the ordering, at least in this
case must have come from the user changing parameter values in the creation of
the `RandomForestRegressor` estimator.


```python
m.feature_importances_
```




    array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.018051  , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.02573276, 0.01708777,
           0.12155166, 0.00740567, 0.49848064, 0.        , 0.        ,
           0.        , 0.        , 0.00679074, 0.        , 0.01795456,
           0.        , 0.        , 0.16862501, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.00226778, 0.11605241, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ])



#### Test Setup - Create 1000 Samples


```python
def fp(train=train, xs=xs, y=y, i=1):
    cc = train.columns.tolist()
    tup = [(c, []) for c in cc]
    msd = dict(tup)
    fpd = dict(tup)
    for t in range(i):
        mi = DecisionTreeRegressor(
            max_features=40, min_samples_leaf=50, random_state=seed
        )
        mi.fit(xs, y)
        fii = mi.feature_importances_
        for c, s in zip(cc, fii):
            fpd[f"{c}"].append(s)
    cc = train.columns.tolist()
    tup = [(c, []) for c in cc]
    msd = dict(tup)
    for i in fpd.keys():
        num = 0
        mean, std = np.mean(fpd[i]), np.std(fpd[i])
        msd[i].append(mean)
        msd[i].append(std)
        if num == 0:
            num = len(fpd[i])

    dffp = pd.DataFrame(msd).T.sort_values(by=0, ascending=False)
    dffpg0 = dffp[dffp.iloc[:, 0] > 0]
    cols_filter = dffpg0.reset_index()
    dffpdo =pd.DataFrame(fpd,columns=cols_filter['index'].tolist())
    fig, ax = plt.subplots(1,2,figsize=(22,10))
    ax=plt.subplot(121)
    dffpg0[0].plot(kind="barh", figsize=(12, 7), legend=False)
    plt.title(f"mean of {num} samples")
    plt.subplots_adjust(left=.14)
    plt.subplots_adjust(bottom=.2)
    plt.ylabel('features')
    plt.xlabel('relative feature importance, if greater 0')
    ax=plt.subplot(122)
    sns.boxplot(data=dffpdo,ax=ax)
    ax.set_title(f'Boxplot Of Feature Importances')
    plt.xticks(rotation=90)
    plt.show()

fpdo = fp(i=1000)
```


    

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_89_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Left: Feature importance plot for features with feature importance score
        greater 0 over 1000 samples.<br>
        Right: Boxplot of 1000 samples, showing no change in rank and almost no
        deviations in scores across samples.
</div>
    


<!-- --- -->
<!-- layout: distill -->
<!-- title: 'Deep Dive Tabular Data Pt. 1' -->
<!-- date: 2023-01-09 -->
<!-- description: 'Preprocssing Data' -->
<!-- img: 'assets/img/838338477938@+-791693336.jpg' -->
<!-- tags: ['tabular data', 'fastai', 'pandas', 'tabular data', 'hypterparameter optimization'] -->
<!-- category: ['tabular data'] -->
<!-- authors: 'Tobias Klein' -->
<!-- comments: true -->
<!-- --- -->
<!-- <br> -->
# Part 3

### RandomForestRegressor (RFR) For Interpretation

#### Create Root Mean Squared Error Metric For Scoring Models

Given, that the kaggle competition we want to submit our final predictions uses
a *root mean squared error* as scoring metric, we create this metric for all
future scoring of predictions.


```python
def r_mse(pred, y):
    return round(math.sqrt(((pred - y) ** 2).mean()), 6)

def m_rmse(m, xs, y):
    return r_mse(m.predict(xs), y)
```

We score the fitted model on the training set, in order to be able to compare it
to the predictions on the validation set. Only the predictions and the score on
the validation dataset matter in the end.


```python
m_rmse(m, xs, y)
```




    0.207714



The model is scored on the validation set.


```python
m_rmse(m, valid_xs, valid_y)
```




    0.198003



#### Custom Function To Create And Fit A RFR Estimator

A function, that creates a `RandomForestRegressor` estimator and fits it using
the training data. Its output is the fitted estimator.


```python
def rf(
    xs,
    y,
    n_estimators=30,
    max_samples=500,
    max_features=0.5,
    min_samples_leaf=5,
    **kwargs,
):
    return RandomForestRegressor(
        n_jobs=-1,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        random_state=seed,
    ).fit(xs, y)
```


```python
m = rf(xs, y)
```

`m_rmse` is used to calculate the mean squared error for the estimator on the
training and validation set.


```python
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
```




    (0.124994, 0.13605)



#### RFR - Theory

The `RandomForestRegressor` model uses a subset of the training set to train
each decision tree in the ensemble on. *The following parameter names are
specific to the implementation in the sklearn library*. The number of decision
trees in the ensemble depends on the value of parameter `n_estimators` and the
size of the subset is controlled by `max_samples`, given the default value of
`bootstrap=True`. If `bootstrap=False` or `max_samples=None`, then each base
estimator is trained using the entire training set. See [*RandomForestRegressor
Documentation*](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

Given that each base base estimator was trained with a maximum of 500 samples or
~43% of all training samples, it is likely that the average of the *rmse* over the predictions
for the samples in the validation dataset is more volatile for a low number of
estimators, and is less volatile as additional base estimators are added one by
one, until all 30 are used and their predictions are averaged. We expect the
value for the average rmse over all 30 base estimators to be equal to the rmse
value we got when executing `m_rmse(m, valid_xs, valid_y)`.


```python
x=len(xs)
pct = np.ceil((500/x)*100)
print(f'Percentage of samples each base estimator uses of the training data: {pct}\n')
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
print(f'The mean rmse over all trees on the validation dataset is: {r_mse(preds.mean(0), valid_y)}')
```

    Percentage of samples each base estimator uses of the training data: 43.0
    
    The mean rmse over all trees on the validation dataset is: 0.13605


#### RFR - Average RMSE By Number Of Estimators

Visualization of the average rmse value, by number of estimators added.


```python
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax = plt.subplot(111)
ax.plot([r_mse(preds[: i + 1].mean(0), valid_y) for i in range(31)])
ax.set_title("Change In RMSE By Estimator")
ax.set_xlabel("Number Of Estimators")
ax.set_ylabel("RMSE")
plt.show()
```


    

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_104_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        RMSE value on the training set, as base estimators are added one by one.
        Value is volatile for low number of estimators, less so as more
        estimators are added.
</div>
    


### Out-Of-Bag Error Explained

In the case, where the size of the subset used to train each base estimator was
smaller than the set of the training data, one can look at the *out-of-bag
error*, as seen below. It is a metric one can use in addition to
*cross-validation* to estimate the models performance on unseen data.

> Out-of-bag error<br>
> The out-of-bag error is the mean error on each training sample, using only the trees
> whose training subset did not include that particular sample.

The value for the rmse, only using out-of-bag samples is higher than the rmse
on the training and validation set, which was expected.


```python
r_mse(m.oob_prediction_, y)
```




    0.154057



### RFR: Standard Deviation Of RMSE By Number Of Estimators

We want to answer the following question:

- How confident are we in our predictions using a particular row of data?

One can visualize the distribution of the standard deviation of the predictions
over all estimators for each sample in the validation dataset. Given, that
estimator was trained on a subset of maximum 500 samples, the standard deviation
can be relatively large for certain estimator sample combinations in the
validation set. The average over all 30 base estimators however generalizes the
final value for each sample in the validation set again and can be interpreted
as the confidence in the predictions of the model for each sample. E.g., a
low/high standard deviation for a particular sample means, that the spread in
the predicted sale price across all estimators is low/high and thus, the
confidence of the model in the prediction is high/low.

#### Visualization Using Swarmplot

The values on the x-axis of the plot are $$\hat{\sigma}_{ln\,Y}$$, with $$Y$$
the random variable, the distribution of the standard deviation over all
estimators for each sample in the validation dataset.


```python
preds_std = np.sort(
    preds.std(0),
    axis=0,
)
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax = plt.subplot(111)
sns.swarmplot(preds_std,orient='h',alpha=0.9,ax=ax)
ax.set_ylabel("Std Of RMSE By OOB Sample ")
ax.set_xlabel("Distribution Of Std")
ax.set_title(f'Swarm Plot Of Standard Deviation (Std)\nOf RMSE On OOB Samples')
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_108_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
     
</div>
    


#### Feature Importances And Selection Using RFR

The dataset used here has 80 independent variables, of which most are assigned
0, which means that their respective contribution, relative to the other
features in terms of lowering the overall *rmse* value is non-existent, judging
by the `feature_importances_` method. The underlying metric is the Gini
importance metric, as mentioned earlier.

##### Compute Feature Importance Scores
A function is created, that calculates the feature importances and returns a
sorted DataFrame.


```python
def rf_feat_importance(m, df):
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}).sort_values("imp", ascending=False)
```

##### Display The Top 10
We call the function and display the 10 features with the highest feature
importance scores.


```python
fi = rf_feat_importance(m, xs)
fi[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>overallqual</td>
      <td>0.336823</td>
    </tr>
    <tr>
      <th>62</th>
      <td>grlivarea</td>
      <td>0.156380</td>
    </tr>
    <tr>
      <th>52</th>
      <td>yearbuilt</td>
      <td>0.056838</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>garageyrblt</td>
      <td>0.038850</td>
    </tr>
    <tr>
      <th>58</th>
      <td>totalbsmtsf</td>
      <td>0.035039</td>
    </tr>
    <tr>
      <th>32</th>
      <td>fireplacequ</td>
      <td>0.021616</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 2 columns</p>
</div>



##### Bar Plot: Feature Importances
Function for the visualization of feature importance scores.


```python
def plot_fi(fi):
    return fi.plot("cols", "imp", "barh", figsize=(12, 7), legend=False,title='Feature Importance Plot',ylabel='Feature',xlabel='Feature Importance Score')
```

##### Display The Top 30
Visualization of the 30 features, with the highest feature importance scores in
ascending order.


```python
plot_fi(fi[:30])
```




    <AxesSubplot: title={'center': 'Feature Importance Plot'}, xlabel='Feature Importance Score', ylabel='Feature'>




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_116_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


##### Create Subset Of Features To Keep

We try to answer the following question:

- Which columns are effectively redundant with each other, for purposes of
prediction?

An arbitrary cutoff value of 0.005 is used, and only column names of features
with a feature importance score larger 0.005 are kept. The number of columns
that remain in the subset is printed.


```python
to_keep = fi[fi.imp > 0.005].cols
len(to_keep)
```




    21



#### New Training Set *xs_imp* After Feature Elimination
`xs_imp`/`valid_xs_imp` is the subset of `xs`/`valid_xs` with the columns, that
are in `to_keep`.


```python
xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]
```

A `RandomForestRegressor` is fitted using `xs_imp`.


```python
m = rf(xs_imp, y)
```

##### RMSE Values Using *xs_imp*

We want to answer this question:

- How do predictions change, as we drop subsets of the features?

The rmse values for the predictions on the training and validation dataset.


```python
m_rmse(m, xs_imp, y), m_rmse(m, valid_xs_imp, valid_y)
```




    (0.13191, 0.139431)



#### Interpretation Of RMSE Values

The rmse values for the smaller feature set are worse compared to the ones using
all features. In this case, this is a problem, given that kaggle scores our
submission only on the rmse value of the test set. In reality, this might be
different. A model has to predict the sale price for houses it has never seen
and not only on the kaggle test set. The predictions might be more robust and
easier to interpret, given a smaller set of features.


```python
len(xs.columns), len(xs_imp.columns)
```




    (83, 21)



As a result of using `xs_imp` instead of `xs` and thus a smaller set of
features, the feature importances can


```python
plot_fi(rf_feat_importance(m, xs_imp))
```




    <AxesSubplot: title={'center': 'Feature Importance Plot'}, xlabel='Feature Importance Score', ylabel='Feature'>




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_128_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


<!-- --- -->
<!-- layout: distill -->
<!-- title: 'Deep Dive Tabular Data Pt. 1' -->
<!-- date: 2023-01-09 -->
<!-- description: 'Preprocssing Data' -->
<!-- img: 'assets/img/838338477938@+-791693336.jpg' -->
<!-- tags: ['tabular data', 'fastai', 'pandas', 'tabular data', 'hypterparameter optimization'] -->
<!-- category: ['tabular data'] -->
<!-- authors: 'Tobias Klein' -->
<!-- comments: true -->
<!-- --- -->
<!-- <br> -->
# Part 4

### Dendrogram Visualization For Spearman Rank Correlations

With the *scipy* library, one can create a dendrogram using the features in the
training set. Using the Spearman rank correlation ($$\rho$$), which does not have any
prerequisites in terms of the relationships between the features, e.g., a linear
relationship, the rank correlation is used to calculate the correlations. The
final plot applies the same color to features whose rank correlation values are
close to each other. On the x-axis, one can see the strength of the correlation,
with close to zero indicating, that the assumed linear relationship between the
features in this group is one of no consistent change. Values close to one
indicate a close to perfect monotonic relationship. $$H_0$$ is, that each variable
pair is uncorrelated.


```python
def cluster_columns(df, figsize=(12, 12), font_size=10):
    i = 'average'
    corr = scipy.stats.spearmanr(df).correlation
    z = hc.linkage(corr, method=i)
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns,
                  orientation="left",distance_sort=True, leaf_font_size=font_size)
    plt.title(f'Dendrogram Of Correlation Clusters\nMethod: "{i}"')
    plt.ylabel('Feature Names By Cluster')
    plt.xlabel(r"Cluster Hierarchy Using Spearman $\rho$")
    plt.show()

cluster_columns(xs_imp)
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_130_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


#### Conclusion

The output shows, that variable `garagetype` and `exterqual` are assumed to have
a high similarity and are the two columns with the strongest assumed linear
relationship among the features found in `xs_imp`.

### Dendrogram Findings Applied

To test what the impact of dropping even more features from `xs_imp`, on the
rmse on the training set is, we look at the *out-of-bag error* (oob). The features
tested are `garagetype` and `exterqual`, since they showed to be the two
features with the highest pair correlation in the training set.


```python
l = dict([(k,[]) for k in ['features','oob_score']])
def get_oob(df,n:str,l=l):
    l['features'].append(n)
    print(l)
    m = RandomForestRegressor(
        n_estimators=30,
        min_samples_leaf=5,
        max_samples=500,
        max_features=0.5,
        n_jobs=-1,
        oob_score=True,
        random_state=seed,
    )
    m.fit(df, y)
    l['oob_score'].append(m.oob_score_)
    print()
    print(f'oob using {n}: {m.oob_score_}')
    print(l)
    return m.oob_score_
```

The baseline oob  score for `xs_imp`.


```python
get_oob(xs_imp,'xs_imp')
```

    {'features': ['xs_imp'], 'oob_score': []}
    
    oob using xs_imp: 0.843435210902987
    {'features': ['xs_imp'], 'oob_score': [0.843435210902987]}





    0.843435210902987



The oob score for `xs_imp` is slightly lower than the one for `xs`. Considering,
that there are 83 columns in `xs` and 21 in `xs_imp`, the slight decrease in the
oob score shows that most of the difference in features between the two is made
up of columns, that don't decrease the rmse of the model on the training data.


```python
get_oob(xs,'xs')
```

    {'features': ['xs_imp', 'xs'], 'oob_score': [0.843435210902987]}
    
    oob using xs: 0.8521572851913823
    {'features': ['xs_imp', 'xs'], 'oob_score': [0.843435210902987, 0.8521572851913823]}





    0.8521572851913823



The two columns `garagetype` and `exterqual` are dropped and the oob score is
computed using the remaining features in `xs_imp`. One notices, that the
oob_score is lower for the case where only one of the two features is dropped
and higher if both are dropped together. Accounting for this, both features are
dropped together.

#### Drop One Feature At A Time
Drop `garagetype` and `exterqual` one at a time, with replacement.


```python
{
    c: get_oob(xs_imp.drop(c, axis=1),f'xs_imp.drop-{c}')
    for c in (
        "garagetype",
        "exterqual",
    )
}
```

    {'features': ['xs_imp', 'xs', 'xs_imp.drop-garagetype'], 'oob_score': [0.843435210902987, 0.8521572851913823]}
    
    oob using xs_imp.drop-garagetype: 0.8447838576955676
    {'features': ['xs_imp', 'xs', 'xs_imp.drop-garagetype'], 'oob_score': [0.843435210902987, 0.8521572851913823, 0.8447838576955676]}
    {'features': ['xs_imp', 'xs', 'xs_imp.drop-garagetype', 'xs_imp.drop-exterqual'], 'oob_score': [0.843435210902987, 0.8521572851913823, 0.8447838576955676]}
    
    oob using xs_imp.drop-exterqual: 0.8443621750779751
    {'features': ['xs_imp', 'xs', 'xs_imp.drop-garagetype', 'xs_imp.drop-exterqual'], 'oob_score': [0.843435210902987, 0.8521572851913823, 0.8447838576955676, 0.8443621750779751]}





    {'garagetype': 0.8447838576955676, 'exterqual': 0.8443621750779751}



#### Drop Both Features Together
Drop both features together and compute the `oob_score`


```python
to_drop = [
    "garagetype",
    "exterqual",
]
get_oob(xs_imp.drop(to_drop, axis=1),f'xs_imp.drop-{to_drop}')
```

    {'features': ['xs_imp', 'xs', 'xs_imp.drop-garagetype', 'xs_imp.drop-exterqual', "xs_imp.drop-['garagetype', 'exterqual']"], 'oob_score': [0.843435210902987, 0.8521572851913823, 0.8447838576955676, 0.8443621750779751]}
    
    oob using xs_imp.drop-['garagetype', 'exterqual']: 0.8500505935232953
    {'features': ['xs_imp', 'xs', 'xs_imp.drop-garagetype', 'xs_imp.drop-exterqual', "xs_imp.drop-['garagetype', 'exterqual']"], 'oob_score': [0.843435210902987, 0.8521572851913823, 0.8447838576955676, 0.8443621750779751, 0.8500505935232953]}





    0.8500505935232953



#### Evaluate *oob_scores*

Create a DataFrame and sort values by `oob_score` in descending order.


```python
df_oob = pd.DataFrame(l).sort_values(by='oob_score',ascending=False)
df_oob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>oob_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>xs</td>
      <td>0.852157</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xs_imp.drop-['garagetype', 'exterqual']</td>
      <td>0.850051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xs_imp.drop-garagetype</td>
      <td>0.844784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xs_imp.drop-exterqual</td>
      <td>0.844362</td>
    </tr>
    <tr>
      <th>0</th>
      <td>xs_imp</td>
      <td>0.843435</td>
    </tr>
  </tbody>
</table>
</div>



### New Train & Validation Sets Using Resulting Feature Set

The resulting Datasets have the two features removed. These two datasets are the
new baseline datasets, that all of the following models are fitted/evaluated on.


```python
xs_final = xs_imp.drop(to_drop, axis=1)
valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)
```

Exporting and immediately importing the datasets in their current state as
*.pkl* files using the fastai proprietary functions `save_pickle` and
`load_pickle` respectively.


```python
save_pickle("xs_final.pkl", xs_final)
save_pickle("valid_xs_final.pkl", valid_xs_final)
```


```python
xs_final = load_pickle("xs_final.pkl")
valid_xs_final = load_pickle("valid_xs_final.pkl")
```

#### Baseline RMSE Scores


```python
m = rf(xs_final, y)
m_rmse(m, xs_final, y), m_rmse(m, valid_xs_final, valid_y)
```




    (0.129697, 0.140712)




```python
dfi = rf_feat_importance(m, xs_final)
plot_fi(dfi) 
```




    <AxesSubplot: title={'center': 'Feature Importance Plot'}, xlabel='Feature Importance Score', ylabel='Feature'>




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_150_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


## Exploring The Impact of Individual Columns

Column `overallqual` has shown to be the most important column overall relative
to all the other columns in the dataset. This gives reason, to get a detailed
look at its unique value distribution. The feature has ten levels, ranging from
two to ten in ascending order. It describes the "overall quality" of an object
and judging by the feature importance plots, it is the strongest predictor for
variable `saleprice`.

The value counts and the box plot for `overallqual` are given below.


```python
xs_final["overallqual"].value_counts().reset_index().sort_values(by='index',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>overallqual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>128</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>18</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 2 columns</p>
</div>




```python
fig, ax = plt.subplots(2,1,figsize=(8,10))
ax= plt.subplot(211)
xs_final["overallqual"].value_counts().plot.barh(ax=ax)
ax.set_xlabel("Absolute Value Counts")
ax.set_ylabel("Unique Values")
ax.set_title(f'Value Distribution for "overallqual"\non the training set')
ax = plt.subplot(212)
sns.boxplot(xs_final["overallqual"],orient='h',ax=ax)
ax.set_xlabel("Unique Values")
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_153_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


Another important feature is `grlivarea`, which gives the area above ground in
square feet.


```python
fig, ax = plt.subplots(2,1,figsize=(10,10))
ax=plt.subplot(211)
sns.histplot(x=xs_final["grlivarea"],ax=ax)
ax.set_xlabel("Values")
ax.set_ylabel("Absolute Frequency")
ax.set_title('Value Distribution for "grlivarea"\non the training set')
ax = plt.subplot(212)
sns.boxplot(xs_final["grlivarea"],orient='h',ax=ax)
plt.title('Boxplot Of "Above Ground Living Area"')
plt.xlabel("Values (Square Feet)")
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_155_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


While the univariate distributions of the features are of interest, they don't
show the relationship between independent and dependent variable. The
relationships can be visualized using a *Partial Dependence* plot.

## Partial Dependence

Plots of partial dependence for the most important columns in the slimmed down
dataset. The plot is part of the *scikit-learn* library and its documentation
can be found here: [*partial_dependence
Documentation*](https://scikit-learn.org/stable/modules/partial_dependence.html).
The plot is like an individual conditional expectation plot, which lets one
calculate the dependence between the dependent variable and any subset of the
independent variables. Four columns, that have shown several times, that they
are of high importance for the predictions of the dependent variable are chosen
and their partial dependence plots are created.

The output shows, that `overallqual` and `yearbuilt` show a high correlation with
the dependent variable. Not only that though, the plot also shows how the
assumed change in the value of the dependent variable, $$\frac{\partial \mathrm{saleprice}}{\partial x_{i}}\,\,\mathrm{for}\,\,\mathrm{i}\, \in \{\mathrm{overallqual},\, \mathrm{grlivarea},\, \mathrm{garagecars},\, \mathrm{yearbuilt}\}$$


```python
ax = plot_partial_dependence(
    m,
    xs_final,
    ["overallqual", "grlivarea", "garagecars", "yearbuilt"],
    grid_resolution=20,
    n_jobs=-1,
    random_state=seed,
    n_cols=4,
)
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_157_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


## Tree Interpreter

A plot using `treeinterpreter` from treeinterpreter and we try to answer the
question:

- For predicting with a particular row of data, what were the most important
factors, and how did they influence that prediction?


```python
row = valid_xs_final.sample(n=2, random_state=seed)
predict, bias, contributions = treeinterpreter.predict(m, row.values)
# rounding for display purposes
predict = np.round(predict,3)
bias = np.round(bias,3)
contributions = np.round(contributions,3)

print(
    f'For the first row in the sample predict is: {predict[0]},\n\nThe overall log average of column "saleprice": {bias[0]},\n\nThe contributions of the columns are:\n\n{contributions[0]}'
)
```

    For the first row in the sample predict is: [12.009],
    
    The overall log average of column "saleprice": 12.022,
    
    The contributions of the columns are:
    
    [ 0.142 -0.081  0.105 -0.016 -0.015 -0.05   0.014  0.026 -0.073 -0.005
     -0.007 -0.035  0.002  0.009  0.    -0.009 -0.001 -0.016 -0.003]



```python
for e in zip(predict, bias, contributions):
    waterfall(
        valid_xs_final.columns,
        e[2],
        Title=f"Predict: {e[0][0]}, Intercept: {e[1]}",
        threshold=0.08,
        rotation_value=45,
        formatting="{:,.3f}",
    )
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_160_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Example of how to use the treeinterpreter on a single sample.
</div>
    



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_160_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Example of how to use the treeinterpreter on a single sample.
</div>
    


<!-- --- -->
<!-- layout: distill -->
<!-- title: 'Deep Dive Tabular Data Pt. 1' -->
<!-- date: 2023-01-09 -->
<!-- description: 'Preprocssing Data' -->
<!-- img: 'assets/img/838338477938@+-791693336.jpg' -->
<!-- tags: ['tabular data', 'fastai', 'pandas', 'tabular data', 'hypterparameter optimization'] -->
<!-- category: ['tabular data'] -->
<!-- authors: 'Tobias Klein' -->
<!-- comments: true -->
<!-- --- -->
<!-- <br> -->
# Part 5: Out-Of-Domain Problem

A series of 45 linear values for the x-axis is created and a corresponding series
of y-values, that projects each x-axis value to its identity and adds some noise
to each projected value using values from sampling a Normal Distribution.


```python
xlins = torch.linspace(0, 20, steps=45)
ylins = xlins + torch.randn_like(xlins)
plt.scatter(xlins, ylins)
```








    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_162_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


There has to be more than one axis, in order for an estimator to be trained on this
data. We can do so by using method `.unsqueeze`.


```python
xslins = xlins.unsqueeze(1)
xlins.shape, xslins.shape
```




    (torch.Size([45]), torch.Size([45, 1]))



Or, one can create the second axis by slicing `xslins` using special variable `None`.


```python
xslins[:, None].shape
```




    torch.Size([45, 1, 1])



Next, train a `RandomForestRegressor` estimator using the first 35 rows of `xslins`
and `ylins` respectively.


```python
m_linrfr = RandomForestRegressor().fit(xslins[:35], ylins[:35])
# Do the same and train a `XGBRegressor` using the same data.
m_lin = XGBRegressor().fit(xslins[:35], ylins[:35])
```

Scatter plot of data points used for training, plus final five values in
`xslins`, that were omitted in the training data using different colored dots
and predicted values for all points in `xslins`. Notice, that the predictions
for the omitted values are all too low. The model can only make predictions
within the range of what it has seen for the dependent variable during training.
This is an example of an extrapolation problem.


```python
plt.scatter(xlins, m_linrfr.predict(xslins), c="r", alpha=0.4,s=10,label='RandomForestRegressor Predicted Values')
plt.scatter(xlins, m_lin.predict(xslins), c="b",marker="2", alpha=0.5,label='XGBRegressor Predicted Values')
plt.scatter(xslins, ylins, 20,marker="H",alpha=0.5, c="y",label='Set Of All Values')
plt.vlines(x=15.5,ymin=0,ymax=20,alpha=0.7,label='Last Independent Training Value')
plt.title('Visualization Of The Out-Of-Domain Problem')
plt.legend(loc='best')
plt.xlabel('x-Axis')
plt.ylabel('y-Axis')
```




    Text(0, 0.5, 'y-Axis')




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_170_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


Why does this happen one might ask. The reason lies within the structure of how
the `RandomForestRegressor` estimator works and the `XGBRegressor` as well in
this regard. All it does is average the predictions of several trees
(`nestimators`) for each sample in the training data. Each tree averages the
values of the dependent variable for all samples in a leaf. This problem can
lead to predictions on so called *out-of-domain data*, that are systematically
too low. One has to ensure, that the validation set does not contain such data.

### Identifying *Out-Of-Domain Data*

**Tool: RandomForestRegressor** The `RandomForestRegressor` is used to predict
whether a row is part of the train or the validation set. For this, the train
and validation data is merged and a new dependent variable, a boolean column, is
added to the dataset. This column indicates whether the value of the dependent
variable is part of the train or validation set.


```python
df_comb = pd.concat([xs_final, valid_xs_final])
valt = np.array([0] * len(xs_final) + [1] * len(valid_xs_final))
m = rf(df_comb, valt)
fi = rf_feat_importance(m, df_comb)
fi.iloc[:5, :]
# #### Baseline rmse
# The rmse value is printed, before dropping any columns.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>lotarea</td>
      <td>0.119988</td>
    </tr>
    <tr>
      <th>1</th>
      <td>grlivarea</td>
      <td>0.108382</td>
    </tr>
    <tr>
      <th>4</th>
      <td>garagearea</td>
      <td>0.098144</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1stflrsf</td>
      <td>0.094516</td>
    </tr>
    <tr>
      <th>8</th>
      <td>totalbsmtsf</td>
      <td>0.092451</td>
    </tr>
  </tbody>
</table>
</div>




```python
m = rf(xs_final, y)
print(f"Original m_rmse value is: {m_rmse(m,valid_xs_final,valid_y)}")
# #### Features To Drop
# The columns, that can be dropped while increasing the accuracy are `garagearea`
# and `garageyrblt`, if only one of them is dropped. There is no information for
# what the accuracy is, if both are dropped together. We test the three cases
# below to decide what the best choice is.
```

    Original m_rmse value is: 0.140712



```python
for c in fi.iloc[:8, 0]:
    m = rf(xs_final.drop(c, axis=1), y)
    print(c, m_rmse(m, valid_xs_final.drop(c, axis=1), valid_y))
```

    lotarea 0.142546
    grlivarea 0.144781
    garagearea 0.137757
    1stflrsf 0.140024
    totalbsmtsf 0.140971
    lotfrontage 0.141214
    bsmtfinsf1 0.143163
    yearremodadd 0.143747


It looks like we can remove `garagearea` without losing any accuracy, which is
confirmed below.


```python
bf = ["garagearea"]
m = rf(xs_final.drop(bf, axis=1), y)
print(m_rmse(m, valid_xs_final.drop(bf, axis=1), valid_y))
```

    0.137757


It looks like we can remove `garageyrblt` as well, without losing any accuracy,
which is confirmed again.


```python
bf = ["garageyrblt"]
m = rf(xs_final.drop(bf, axis=1), y)
print(m_rmse(m, valid_xs_final.drop(bf, axis=1), valid_y))
```

    0.138856


Dropping both decreases the accuracy on the validation set however and so only
column `garagearea` is dropped.


```python
bf = ["garageyrblt", "garagearea"]
m = rf(xs_final.drop(bf, axis=1), y)
print(m_rmse(m, valid_xs_final.drop(bf, axis=1), valid_y))
```

    0.14009


It is confirmed, that using the current value for `random_state`, the
independent variable `garagearea` can be dropped while the `m_rmse` value even
decreases on the test set by doing so. Therefore, the independent part of the
train and validation data is updated, variable `garagearea` is dropped in both.


```python
bf = ["garagearea"]
xs_final_ext = xs_final.drop(bf, axis=1)
valid_xs_final_ext = valid_xs_final.drop(bf, axis=1)
```

Writing down the column names, so executing previous cells again later will not
change which columns are part of `xs_final_ext` and `valid_xs_final_ext`
DataFrames used to determine which columns are part of the DataFrames
constructed using these columns.


```python
for i in ["xs_final_ext", "valid_xs_final_ext"]:
    pd.to_pickle(i, f"{i}.pkl")

finalcolsdict = {
    "xs_final_ext": xs_final_ext.columns.tolist(),
    "valid_xs_final_ext": valid_xs_final_ext.columns.tolist(),
}
```

We look at a scatter plot of the data in `garagearea` and the distributions of
its values across the train and validation set. From the plot, it becomes
apparent that the distribution of `garagearea`, while having a close to
identical median for both datasets, has a lower Q1 among other differences. Some
of the values larger than the Q4 upper bound for the training set are larger,
than the ones found in the validation set.


```python
plt.close("all")
fig = plt.figure(figsize=(6,6))
sns.histplot(data=df_comb, x="lotarea", hue=valt,stat='density',multiple='stack',)
plt.title('Histogram Of "lotarea"\n0 == training data, 1 == validation data')
plt.ylabel('Relative Frequency')
plt.xlabel('Value (Square Feet)')
sns.displot(data=df_comb, x="lotarea", hue=valt,kind='ecdf')
plt.title('ECDF Of "lotarea"\n0 == training data, 1 == validation data')
plt.ylabel('Cumulative Density')
plt.xlabel('Value (Square Feet)')
plt.subplots_adjust(top=.9)
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_186_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_187_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        ECDF plot is not rendered well, as it is not yet implemented in
        matplotlib, according to an error message.
</div>
    


<!-- --- -->
<!-- layout: distill -->
<!-- title: 'Deep Dive Tabular Data Pt. 1' -->
<!-- date: 2023-01-09 -->
<!-- description: 'Preprocssing Data' -->
<!-- img: 'assets/img/838338477938@+-791693336.jpg' -->
<!-- tags: ['tabular data', 'fastai', 'pandas', 'tabular data', 'hypterparameter optimization'] -->
<!-- category: ['tabular data'] -->
<!-- authors: 'Tobias Klein' -->
<!-- comments: true -->
<!-- --- -->
<!-- <br> -->
# Part 6: Creation Of The Kaggle Submission

For the final submission, we train several models and combine their predictions
in the form of a weighted ensemble prediction. Estimators from the following
model types are included. Number of iterations marks the final number of
iterations used for the submission.

- `RandomForestRegressor` from *sklearn*
- `XGBRegressor` from *xgboost*
- `tabular_learner` (deep learning model) from *fastai*

The hyperparameter optimization for each of them is:

- `RandomForestRegressor`
    - Manual loop with 30 iterations using parameters:
        - `nestimators` - number of estimators to use.
        - `max_samples` - maximum number of samples to use for training a single
           base estimator (tree).
- `tabular_learner`
    - Manual loop with 20 iterations using parameters:
        - `lr` (learning rate) - values tested depend on `lr_find` output.
        - `epochs` Number of epochs to train.
- `XGBRegressor`
    - `RandomizedSearchCV` with 1400 iterations and 8 fold cross-validation for each from *sklearn* using a parameter distribution dictionary.
    - For details, see section 'XGBRegressor Optimization'.

### Creating Estimators Optimized For Kaggle

So far, the focus has been on fitting estimators for interpretability and not
for the lowest rmse value. The kaggle competition we want to submit our final
predictions to however only scores each submission based on rmse value on the
test set and nothing else. This makes it necessary, that we try to create
estimators, that are the result of hyperparameter tuning, starting with few
iterations where we check the resulting rmse values and building up to using as
many iterations, that our hardware can handle within a reasonable duration of no
more than 5 minutes give or take or stop adding more iterations to the hyper
parameter optimization procedure, if rmse values stop improving despite
increasing the number of iterations.

### RandomForestRegressor Optimization

Using a manually created test harness, the rmse values for each iteration on the
training and validation set are appended to list `m_rmsel` and `m_rmselv`
respectively and it is these lists, that are returned by the function.


```python
def rf2(
    xs_final=xs_final,
    y=y,
    valid_xs_final=valid_xs_final,
    valid_y=valid_y,
    nestimators=[60, 50, 40, 30, 20],
    max_samples=[200, 300, 400, 500, 600, 700],
    max_features=0.5,
    min_samples_leaf=5,
    **kwargs,
):
    from itertools import product

    m_rmsel = []
    m_rmselv = []
    setups = product(nestimators, max_samples)
    for ne in setups:
        mt = RandomForestRegressor(
            n_jobs=-1,
            n_estimators=ne[0],
            max_samples=ne[1],
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            oob_score=True,
            random_state=seed,
        ).fit(xs_final, y)
        m_rmsel.append((m_rmse(mt, xs_final, y), ne[0], ne[1]))
        m_rmselv.append((m_rmse(mt, valid_xs_final, valid_y), ne[0], ne[1]))
    return m_rmsel, m_rmselv
```

We run the manual hyperparameter optimization and assign the outputs to
`m_rmset` and `m_rmsev` respectively.


```python
m_rmset, m_rmsev = rf2()
```

The evaluation is done by creating a DataFrame and then using *pandas*
`.groupby` method along with aggregation method `.agg` where we aggregate by the
minimum over each `m_rmsev` value. We choose the parameter combination found in
the first row of the resulting `grouped_opt` DataFrame.


```python
dfm_rmsev = pd.DataFrame(m_rmsev, columns=["m_rmsev", "n_estimators", "max_samples"])
grouped_opt = dfm_rmsev.groupby(by="m_rmsev").agg(min)
grouped_opt.iloc[:5, :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_estimators</th>
      <th>max_samples</th>
    </tr>
    <tr>
      <th>m_rmsev</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.138596</th>
      <td>60</td>
      <td>600</td>
    </tr>
    <tr>
      <th>0.139147</th>
      <td>50</td>
      <td>600</td>
    </tr>
    <tr>
      <th>0.139720</th>
      <td>40</td>
      <td>600</td>
    </tr>
    <tr>
      <th>0.140007</th>
      <td>60</td>
      <td>700</td>
    </tr>
    <tr>
      <th>0.140081</th>
      <td>30</td>
      <td>600</td>
    </tr>
  </tbody>
</table>
</div>



To avoid using the wrong parameter combination, one that is not the optimal one
for the given execution of the code, we assign the values for the optimal number
of `n_estimators` and `max_samples` directly by the index values, that hold the
optimal parameter values in of `grouped_opt`.

Function `rff` will fit a `RandomForestRegressor` with the optimal parameter
values, as found by the hyperparameter optimization procedure outlined above
regardless of execution number.


```python
def rff(
    xs,
    y,
    n_estimators=grouped_opt.iloc[0, 0],
    max_samples=grouped_opt.iloc[0, 1],
    max_features=0.5,
    min_samples_leaf=5,
    **kwargs,
):
    return RandomForestRegressor(
        n_jobs=-1,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        random_state=seed,
    ).fit(xs, y)
```

#### Final RandomForestRegressor rmse Values

Executing function `rff` we get the rmse values for the fitted estimator.


```python
m = rff(xs_final, y)
m_rmse(m, xs_final, y), m_rmse(m, valid_xs_final, valid_y)
```




    (0.124334, 0.138596)



### tabular_learner - Deep Learning Model

While dropping `garagearea` resulted in a slightly higher accuracy using
`RandomForestRegressor` on the validation set, the increase was marginal. Let's
see what the results are using neural networks.

The original csv files are imported and we show how to apply the preprocessing
steps using the `TabularPandas` function from the *fastai* library.

Creating the DataFrames for fitting the deep learning model.


```python
nn_t = base + "/" + "my_competitions/kaggle_competition_house_prices/data/train.csv"
nn_v = base + "/" + "my_competitions/kaggle_competition_house_prices/data/test.csv"
dfnn_t = pd.read_csv(nn_t, low_memory=False).clean_names()
dfnn_v = pd.read_csv(nn_v, low_memory=False).clean_names()
print(len(dfnn_v))
dfnn_v.columns[:3]
```

    1459





    Index(['id', 'mssubclass', 'mszoning'], dtype='object')



Assigning the ordered categorical columns to the data, as we did before for the
tree based models in a previous part. See [**Deep Dive Tabular Data Part 1**]({% link _projects/tabular_kaggle-1.md %})<br>


```python
dfnn_t = cu(dfnn_t, uset, usetna)
dfnn_v = cu(dfnn_v, uset, usetna)
dfnn_t = tl(dfnn_t)
```

    extercond Index(['Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    heatingqc Index(['Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    fireplacequ Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    garagequal Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    garagecond Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    extercond Index(['Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    heatingqc Index(['Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    fireplacequ Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')
    garagecond Index(['FM', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype='object')


Applying the `log` function to the dependent variable `saleprice`.

Only use the columns, that were left in the dataset after analyzing the
contribution of each of the columns in the previous section.


```python
dfnn_tf = dfnn_t[
    xs_final_ext.columns.tolist() + ["saleprice"]
]  # _tf stands for train and final (train dataset from kaggle)
dfnn_vf = dfnn_v[
    xs_final_ext.columns.tolist()
]  # _vf stands for validation final (test dataset from kaggle)
print(len(dfnn_vf))
dfnn_tf.sample(n=3, random_state=seed)
```

    1459





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>overallqual</th>
      <th>grlivarea</th>
      <th>yearbuilt</th>
      <th>garagecars</th>
      <th>1stflrsf</th>
      <th>...</th>
      <th>lotfrontage</th>
      <th>fireplaces</th>
      <th>2ndflrsf</th>
      <th>totrmsabvgrd</th>
      <th>saleprice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>6</td>
      <td>1068</td>
      <td>1963</td>
      <td>1</td>
      <td>1068</td>
      <td>...</td>
      <td>70.0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>11.947949</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>8</td>
      <td>2622</td>
      <td>1994</td>
      <td>2</td>
      <td>1500</td>
      <td>...</td>
      <td>98.0</td>
      <td>2</td>
      <td>1122</td>
      <td>9</td>
      <td>12.691580</td>
    </tr>
    <tr>
      <th>413</th>
      <td>5</td>
      <td>1028</td>
      <td>1927</td>
      <td>2</td>
      <td>1028</td>
      <td>...</td>
      <td>56.0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>11.652687</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 19 columns</p>
</div>



Verify, that the number of columns in `dfnn_tf` is correct.


```python
len(dfnn_tf.columns)
```




    19



#### Testing Of Different Values For Parameter max_card

Values in the range between 2 and 100 are tested. Output is hidden, for
readability.


```python
for i in range(2, 101):
    contnn, catnn = cont_cat_split(dfnn_tf, max_card=i, dep_var="saleprice")
#    print(f"{len(contnn)}, {i}: {contnn}")
```

Looking at the above output, and the fact that it is hard to find a column in
the dataset, that can be clearly identified as having continuous values, only
columns with more than 100 unique values are assigned as being continuous. The
final continuous columns are printed below. The output has the format.

```
(x,y,z)
```

**x := Number of type continuous columns, given threshold value *y***<br>
**y := Minimum for number of unique values, for a column to be assigned type continuous**<br>
**z := List of names of columns assigned type continuous**<br>

Example given below:

```pycon
>>> 9, 100: ['grlivarea', 'yearbuilt', '1stflrsf', 'garageyrblt', 'totalbsmtsf',
                'bsmtfinsf1', 'lotarea', 'lotfrontage', '2ndflrsf']
```

Creating and displaying the continuous and categorical columns using `max_card`
100.


```python
contnn, catnn = cont_cat_split(dfnn_tf, max_card=100, dep_var="saleprice")
catnn
```




    ['overallqual',
     'garagecars',
     'fullbath',
     'fireplacequ',
     'centralair',
     'yearremodadd',
     'garagecond',
     'fireplaces',
     'totrmsabvgrd']




```python
contnn
```




    ['grlivarea',
     'yearbuilt',
     '1stflrsf',
     'garageyrblt',
     'totalbsmtsf',
     'bsmtfinsf1',
     'lotarea',
     'lotfrontage',
     '2ndflrsf']



Print the number of unique values for all columns part of subset categorical
columns.


```python
dfnn_tf[catnn].nunique().sort_values(ascending=False)
```




    yearremodadd    61
    totrmsabvgrd    12
    overallqual     10
                    ..
    fullbath         4
    fireplaces       4
    centralair       2
    Length: 9, dtype: int64



#### Run TabularPandas Function
Since none of the boolean columns, that indicate whether there was or wasn't a
missing value in a row of a column are present in the final training dataset, we
drop these columns from the created tabular object below. Doing this now, helps
us in making the training and test data compatible, if the test data has missing
values in columns, where the training data doesn't.


```python
procsnn = [Categorify, FillMissing(add_col=False), Normalize]
tonn = TabularPandas(
    dfnn_tf,
    procsnn,
    catnn,
    contnn,
    splits=(train_s, valid_s),
    y_names="saleprice",
)
```

#### Create Dataloaders Object

The dataloaders object holds all training and validation sets with the
preprocessed TabularPandas object as input.


```python
dls = tonn.dataloaders(1024)
x_nnt, y = dls.train.xs, dls.train.y
x_val_nnt, y_val = dls.valid.xs, dls.valid.y
y.min(), y.max()
```




    (10.46024227142334, 13.534473419189453)



Calculate the rmse value using the data sets from the dataloaders function.


```python
m2 = rff(x_nnt, y)
m_rmse(m2, x_nnt, y), m_rmse(m2, x_val_nnt, y_val)
```




    (0.124612, 0.135281)



#### Create tabular_learner estimator

Create the `tabular_learner` object using the dataloaders object from
the previous step. The range of the independent variable `saleprice` is adjusted
to be narrower than the default range.


```python
learn = tabular_learner(dls, y_range=(10.45, 13.55), n_out=1, loss_func=F.mse_loss)
```

### Preprocessing Of The Kaggle Test Dataset
A look at the columns of the DataFrame that holds the independent variables,
as given by the Kaggle test dataset. This is the dataset that the final
predictions need to be made on.


```python
dfnn_vf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 18 columns):
     #   Column        Non-Null Count  Dtype   
    ---  ------        --------------  -----   
     0   overallqual   1459 non-null   int64   
     1   grlivarea     1459 non-null   int64   
     2   yearbuilt     1459 non-null   int64   
     3   garagecars    1458 non-null   float64 
     4   1stflrsf      1459 non-null   int64   
     5   fullbath      1459 non-null   int64   
     6   garageyrblt   1381 non-null   float64 
     7   totalbsmtsf   1458 non-null   float64 
     8   fireplacequ   1459 non-null   category
     9   bsmtfinsf1    1458 non-null   float64 
     10  lotarea       1459 non-null   int64   
     11  centralair    1459 non-null   object  
     12  yearremodadd  1459 non-null   int64   
     13  garagecond    1459 non-null   category
     14  lotfrontage   1232 non-null   float64 
     15  fireplaces    1459 non-null   int64   
     16  2ndflrsf      1459 non-null   int64   
     17  totrmsabvgrd  1459 non-null   int64   
    dtypes: category(2), float64(5), int64(10), object(1)
    memory usage: 185.8+ KB


Looking at the first 5 rows.


```python
dfnn_vf.sample(n=5, random_state=seed)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>overallqual</th>
      <th>grlivarea</th>
      <th>yearbuilt</th>
      <th>garagecars</th>
      <th>1stflrsf</th>
      <th>...</th>
      <th>garagecond</th>
      <th>lotfrontage</th>
      <th>fireplaces</th>
      <th>2ndflrsf</th>
      <th>totrmsabvgrd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1321</th>
      <td>4</td>
      <td>864</td>
      <td>1950</td>
      <td>1.0</td>
      <td>864</td>
      <td>...</td>
      <td>TA</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>836</th>
      <td>8</td>
      <td>2100</td>
      <td>2007</td>
      <td>3.0</td>
      <td>958</td>
      <td>...</td>
      <td>TA</td>
      <td>82.0</td>
      <td>2</td>
      <td>1142</td>
      <td>8</td>
    </tr>
    <tr>
      <th>413</th>
      <td>5</td>
      <td>990</td>
      <td>1994</td>
      <td>1.0</td>
      <td>990</td>
      <td>...</td>
      <td>TA</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>522</th>
      <td>8</td>
      <td>1342</td>
      <td>2006</td>
      <td>2.0</td>
      <td>1342</td>
      <td>...</td>
      <td>TA</td>
      <td>48.0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>6</td>
      <td>2422</td>
      <td>1954</td>
      <td>2.0</td>
      <td>2422</td>
      <td>...</td>
      <td>TA</td>
      <td>102.0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 18 columns</p>
</div>



We apply the same procs we used for the training dataset during the call to
`TabularPandas`, followed by creating the dataloaders object and assigning the
independent variables to variable `x_valid`.

Since there is no dependent variable in this dataset, there is no `.y` part. We
omitted the parameter `y_names` for that reason and not passing the function a
value for `splits` does not split the dataset into training and validation data.
All rows will be part of the `dlsv.train.xs` part.

In order to get predictions using the test data from Kaggle using the fitted
estimator, we call the name of the TabularPandas object used for training and
apply method `.new` to it and pass it the *training* data (it is the Kaggle test
data) from dataloaders object `dlsv` by writing `dlsv.train.xs`. The data is
processed and the dataloaders object with the test data is loaded for
predictions.


```python
procsnn = [Categorify, FillMissing(add_col=False), Normalize]
tonn_vf = TabularPandas(dfnn_vf, procsnn, catnn, contnn)
dlsv = tonn_vf.dataloaders(1024)
x_valid = dlsv.train.xs
tonn_vfs = tonn.new(dlsv.train.xs)
tonn_vfs.process()
tonn_vfs.items.head()
tonn_vfs_dl = dls.valid.new(tonn_vfs)
tonn_vfs_dl.show_batch()
```


<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>overallqual</th>
      <th>garagecars</th>
      <th>fullbath</th>
      <th>fireplacequ</th>
      <th>centralair</th>
      <th>yearremodadd</th>
      <th>garagecond</th>
      <th>fireplaces</th>
      <th>totrmsabvgrd</th>
      <th>grlivarea</th>
      <th>yearbuilt</th>
      <th>1stflrsf</th>
      <th>garageyrblt</th>
      <th>totalbsmtsf</th>
      <th>bsmtfinsf1</th>
      <th>lotarea</th>
      <th>lotfrontage</th>
      <th>2ndflrsf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>1</td>
      <td>3</td>
      <td>-1.215545</td>
      <td>-0.340839</td>
      <td>-0.654589</td>
      <td>-0.653021</td>
      <td>-0.370719</td>
      <td>0.063433</td>
      <td>0.363854</td>
      <td>0.567329</td>
      <td>-0.775266</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>1</td>
      <td>4</td>
      <td>-0.323504</td>
      <td>-0.439751</td>
      <td>0.433328</td>
      <td>-0.769719</td>
      <td>0.639182</td>
      <td>1.063523</td>
      <td>0.897789</td>
      <td>0.615966</td>
      <td>-0.775266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>2</td>
      <td>4</td>
      <td>0.294491</td>
      <td>0.844006</td>
      <td>-0.574169</td>
      <td>0.747364</td>
      <td>-0.266790</td>
      <td>0.773368</td>
      <td>0.809397</td>
      <td>0.275533</td>
      <td>0.891941</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>2</td>
      <td>5</td>
      <td>0.243012</td>
      <td>0.876977</td>
      <td>-0.579218</td>
      <td>0.786203</td>
      <td>-0.271295</td>
      <td>0.357968</td>
      <td>0.031786</td>
      <td>0.470066</td>
      <td>0.837236</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>1</td>
      <td>3</td>
      <td>-0.424442</td>
      <td>0.679386</td>
      <td>0.310173</td>
      <td>0.552805</td>
      <td>0.528496</td>
      <td>-0.387169</td>
      <td>-0.971584</td>
      <td>-1.232092</td>
      <td>-0.775266</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>2</td>
      <td>5</td>
      <td>0.348114</td>
      <td>0.712356</td>
      <td>-0.988706</td>
      <td>0.591644</td>
      <td>-0.639604</td>
      <td>-0.965220</td>
      <td>0.036564</td>
      <td>0.324165</td>
      <td>1.346189</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>1</td>
      <td>4</td>
      <td>-0.616098</td>
      <td>0.679386</td>
      <td>0.076580</td>
      <td>0.552805</td>
      <td>0.275484</td>
      <td>1.089883</td>
      <td>-0.370756</td>
      <td>-0.064899</td>
      <td>-0.775266</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>2</td>
      <td>5</td>
      <td>-0.043400</td>
      <td>0.876977</td>
      <td>-0.923342</td>
      <td>0.786203</td>
      <td>-0.580829</td>
      <td>-0.965220</td>
      <td>-0.285948</td>
      <td>-0.259431</td>
      <td>0.832477</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>2</td>
      <td>3</td>
      <td>-0.298774</td>
      <td>0.613678</td>
      <td>0.463439</td>
      <td>0.474945</td>
      <td>0.573757</td>
      <td>0.434900</td>
      <td>0.072399</td>
      <td>0.810498</td>
      <td>-0.775266</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>#na#</td>
      <td>1</td>
      <td>2</td>
      <td>-1.244439</td>
      <td>-0.044803</td>
      <td>-0.689749</td>
      <td>-0.302924</td>
      <td>-0.370719</td>
      <td>0.801959</td>
      <td>-0.285948</td>
      <td>0.081001</td>
      <td>-0.775266</td>
    </tr>
  </tbody>
</table>


# Part 7: Optimization Routines & Final Submission

### tabular_learner Optimization

With the dataloaders objects created for training, as well as for the final
predictions the `tabular_learner` has to be optimized using a manual
hyperparameter optimization routine. Given, that this dataset is relatively
small with less than 2000 rows, a 10 core CPU machine enough for the
optimization.

The initial step is to call method `lr_find` on the `tabular_learner` object, so
the value range for parameter `lr` can be specified. Since the training set is
small, overfitting occurred quickly and a small value for `lr` showed to work
best in this case. Values in list `linspace` were tested. Epochs were tested in
increments of 5 between 20 and 40. Early testing showed that this range produces
relatively consistent rmse values on the validation set.


```python
learn.lr_find()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    SuggestedLRs(valley=0.0003981071640737355)




    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_227_3.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    



```python
linspace_rmse = {"lr": [], "epochs": [], "rmse": []}
preds_targs = {"nn_preds": [], "nn_targs": []}
linspace = [*np.linspace(0.000007, 0.1, 4)]
epochs = [*np.arange(20, 40, 5)]
setups = product(linspace, epochs)
for i in setups:
    linspace_rmse["lr"].append(float(i[0]))
    linspace_rmse["epochs"].append(i[1])
    with learn.no_logging():
        learn.fit(i[1], float(i[0]))
    preds, targs = learn.get_preds()
    preds_targs["nn_preds"].append(preds)
    preds_targs["nn_targs"].append(targs)
    linspace_rmse["rmse"].append(r_mse(preds, targs))
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







Final values for the tested parameters, that result in the lowest rmse value on
the training data are found using the code below.


```python
dfopt = pd.DataFrame(linspace_rmse).sort_values(by="rmse").iloc[:5]
preds_index = dfopt.index[0]
preds = preds_targs["nn_preds"][preds_index]
print(r_mse(preds_targs["nn_preds"][preds_index], preds_targs["nn_targs"][preds_index]))
dfopt
```

    0.132921





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lr</th>
      <th>epochs</th>
      <th>rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0.033338</td>
      <td>30</td>
      <td>0.132921</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.033338</td>
      <td>35</td>
      <td>0.140671</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.066669</td>
      <td>25</td>
      <td>0.151811</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.033338</td>
      <td>25</td>
      <td>0.159139</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.066669</td>
      <td>20</td>
      <td>0.161941</td>
    </tr>
  </tbody>
</table>
</div>



Using an ensemble, consisting of the `tabular_learner` and the optimized
`RandomForestRegressor` estimator is evaluated below, accounting for the
consistently better predictions of the `RandomForestRegressor` compared to the
ones of the `tabular_learner`, the predictions of the `RandomForestRegressor`
are weighted twice as much as the ones from the `tabular_learner`.

The general idea is that different types of models have different rows for which
their predictions are superior to those of the other models in the ensemble, and
therefore an ensemble of models can potentially achieve better predictions over
the entire set of rows compared to the predictions of the individual models in
the ensemble.

This simple ensemble resulted in the lowest rmse value recorded so far on the
validation set.


```python
preds2, _ = learn.get_preds(dl=tonn_vfs_dl)
rf_preds = m2.predict(x_val_nnt)
ens_preds = (to_np(preds.squeeze()) + 2 * rf_preds) / 3
r_mse(ens_preds, y_val)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    0.126166



### XGBRegressor Optimization

The `XGBRegressor` is a top contender when it comes to a tree ensemble based
model, that can deliver predictions of the highest accuracy, having being used in many
competition winning submissions on Kaggle and other platforms in the past.

In order to get the most out of this model, one has to use hyperparameter tuning
during the fitting process together with strong cross-validation techniques to
keep the estimator from overfitting on the training data during the optimization
process.

The parameters used here during the hyperparameter optimization procedure are
the ones used in the *dmlc XGBoost* version of the model. The full list of
parameters, as well as descriptions for each one can be found in the [*dmlc
XGBoost Parameter
Documentation*](https://xgboost.readthedocs.io/en/stable/parameter.html). The
code below uses the *Scikit-Learn API*, found in the docs as well([*Scikit-Learn
XGBoost API
Documentation*](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn))

Functions created for the hyperparameter optimization are `RMSE` and
`get_truncated_normal`.

`get_truncated_normal` is a custom distribution based on the normal
distribution, that ensures that none of its values are below or above a
specified value. Between the upper and lower limit, one can adjust the shape of
the bell curve like value distribution.

`RMSE` defines a custom scorer metric, that *sklearn* can use during training to
evaluate the accuracy of the predictions made by the estimator during the
optimization process. It is the metric, that Kaggle uses to evaluate the final
submission.

The hyperparameter optimization used is `RandomizedSearchCV` from the *sklearn*
library. It is a variation of the general method called *random search*. It
shares some characteristics with method *grid search*, but differs in a few key
ascpects from it.

A grid search is a method where the user passes a grid with an exhaustive set of
values to be tested to the algorithm as input. A set of values is passed on as
input for the grid search algorithm, for each parameter to be optimised during
the grid search. The underlying problem with the grid search is, that each
hyperparameter can have a large value range for its possible values. An example
is a parameter with a continuous value range between $$0$$ and $$1$$. This range
containing all $$\textbf{machine numbers}$$ between $$0$$ and $$1$$ could not be
tested in a grid search, as there are too many values in this range. Oftentimes
only a small subset of all hyperparameters in a model and only a small subset of
the respective value ranges for each parameter are of relevance for the value of
the evaluation metric.
However, the number of models can still become extremely high. Consider $$15$$
hyperparameters and for illustration purposes assume each one has $$40$$
possible values. If a $$5$$ fold cross validation is used to evaluate the
models, the total number of models to build is given by $$15 \cdot 40 \cdot 5 =
3000$$. To put it into perspective, with an estimated time of $$0.68$$ seconds
that it takes to build one model on the here used machine, to build all $$3000$$
models would take $$3000 \cdot 0.68 = 34 \;\mathrm{minutes}$$. While more
computational power in the form of better hardware is a solution up to a certain
point, using one of the following methods can be comparably more efficient.
Considering these weaknesses of the grid search procedure, there are
alternatives available and given the number and value range for each parameter
included in the hyperparameter optimization, a random search is chosen over a
grid search.

Random search samples from the distribution passed for each parameter (specified
by `get_truncated_normal`) during each iteration and can therefore cover a wider
range of values for each parameter, while using increments as small as the
number of iterations passed allow it to use for sampling.

1400 iterations are used, each one using a 8 fold cross-validation and
`enable_categorical=True` is passed. The tree method is the default. At the end,
the best estimator is assigned to variable `best` and the rmse on the validation
set is printed out using estimator `best`.


```python
def RMSE(preds_indep_test, dep_test):
    return np.sqrt(mean_squared_error(preds_indep_test, dep_test))

def get_truncated_normal(mean, sd, low, upp):
    return scipy.stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )

model = XGBRegressor(
    random_state=seed, tree_method="hist", nthread=-1, enable_categorical=True
)

param_dist = {
    "n_estimators": scipy.stats.randint(100, 1000),
    "learning_rate": np.random.random_sample(1000),
    "max_depth": scipy.stats.randint(2, 30),
    "subsample": get_truncated_normal(mean=0.6, sd=0.2, low=0, upp=1),
    "colsample_bylevel": get_truncated_normal(mean=0.6, sd=0.4, low=0, upp=1),
    "colsample_bytree": get_truncated_normal(mean=0.6, sd=0.4, low=0, upp=1),
    "colsample_bynode": np.random.choice(
        np.arange(0.1, 1, 0.1), size=1000, replace=True
    ),
    "lambda": np.random.choice(np.arange(0.1, 1.2, 0.01), size=1000, replace=True),
    "alpha": np.random.choice(np.arange(0.1, 1, 0.04), size=100, replace=True),
    "gamma": scipy.stats.expon.rvs(size=1500),
}
n_iter = 1400
kfold = KFold(n_splits=8)
random_search = RandomizedSearchCV(
    model, param_dist, scoring=None, refit=True, n_jobs=-1, cv=kfold, n_iter=n_iter
)
random_result = random_search.fit(x_nnt, y)
best = random_result.best_estimator_
random_proba = best.predict(x_val_nnt)
val_score = RMSE(random_proba, y_val)
print(f"{val_score} val_score")
```

    0.13291718065738678 val_score


### Three Model Ensemble
An ensemble consisting of `tabular_learner`, `RandomForestRegressor` and
`XGBRegressor` is tested using equal weights for each one. The results beat the
previous ones and this ensemble is used in the final submission.


```python
ens_preds2 = (to_np(preds.squeeze()) + rf_preds + random_proba) / 3
r_mse(ens_preds2, y_val)
```




    0.124315



### Kaggle Submission

Function `val_pred` takes a dataloader and the three estimators optimized in the
previous sections as inputs and 

Since the rmse of the predictions is lowest when only using `XGBRegressor` and
`RandomForestRegressor`, only these two are used for the final predictions
submitted to Kaggle.

The exponential function is applied to the predictions of each estimator before
adding them together and dividing them by three. This re-transforms them. In this
form the predictions can be added to a DataFrame under a new column `SalePrice`
and exported as CSV file.

Finally, since this submission turned out to be worse than the best one I had
submitted prior, the predictions of the best one to that point was imported and
the bivariate distribution of the dependent variable was plotted and compared.
The best submission is *9* and so `SalePrice09` gives the predictions of this
submission.


```python
def val_pred(tonn_vfs_dl,m2,best,learn):
    predsnn, _ = learn.get_preds(dl=tonn_vfs_dl)
    predsrf = m2.predict(tonn_vfs_dl.items)
    predsxgb = best.predict(tonn_vfs_dl.items)
    for i in predsnn,predsrf,predsxgb:
        print(type(i),len(i),i.shape)
    preds_ens_final = (np.exp(to_np(predsnn.squeeze())) + np.exp(predsrf) + np.exp(predsxgb)) /3
    print(preds_ens_final.shape,preds_ens_final[:4])
    df_sub=pd.read_csv('/Users/tobias/all_code/projects/python_projects/kaggle/my_competitions/kaggle_competition_house_prices/data/sample_submission.csv')
    df_sub9=pd.read_csv('/Users/tobias/all_code/projects/python_projects/kaggle/my_competitions/kaggle_competition_house_prices/data/submission_9.csv')
    df_sub9 = (
        df_sub9
        .rename_column('SalePrice','SalePrice09')
    )
    df_sub['SalePrice'] = preds_ens_final
    print(df_sub.sample(n=3))
    df_sub.to_csv('../portfolio_articles/submission_portfolio.csv',index=False)
    df_sub_comp = pd.concat([df_sub[['SalePrice']],df_sub9[['SalePrice09']]],axis=1)
    df_sub_comp['diff'] = np.abs(df_sub_comp['SalePrice']-df_sub_comp['SalePrice09'])
    df_sub_comp['diff'].hist()
    plt.show()
    df_sub_comp = (
        df_sub_comp
        .add_columns(
            predsrf=np.exp(predsrf), predsxgb=np.exp(predsxgb)
        )
    )
    return df_sub_comp

df_sub_comp = val_pred(tonn_vfs_dl,m2,best,learn)
print(df_sub_comp.columns)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    <class 'torch.Tensor'> 1459 torch.Size([1459, 1])
    <class 'numpy.ndarray'> 1459 (1459,)
    <class 'numpy.ndarray'> 1459 (1459,)
    (1459,) [55746.34455882 59377.90043929 57786.0567984  62267.60988243]
            Id     SalePrice
    173   1634  62267.609882
    1213  2674  74203.475024
    799   2260  61325.784554



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_238_3.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    


    Index(['SalePrice', 'SalePrice09', 'diff', 'predsrf', 'predsxgb'], dtype='object')



```python
sns.displot(data=df_sub_comp,x='SalePrice',y='SalePrice09')
plt.show()
sns.kdeplot(data=df_sub_comp,multiple="stack")
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_239_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_239_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    

Entire Series:

[**Deep Dive Tabular Data Part 1**]({% link _projects/tabular_kaggle-1.md %})<br>
[**Deep Dive Tabular Data Part 2**]({% link _projects/tabular_kaggle-2.md %})<br>
[**Deep Dive Tabular Data Part 3**]({% link _projects/tabular_kaggle-3.md %})<br>
[**Deep Dive Tabular Data Part 4**]({% link _projects/tabular_kaggle-4.md %})<br>
[**Deep Dive Tabular Data Part 5**]({% link _projects/tabular_kaggle-5.md %})<br>
[**Deep Dive Tabular Data Part 6**]({% link _projects/tabular_kaggle-6.md %})<br>
[**Deep Dive Tabular Data Part 7**]({% link _projects/tabular_kaggle-7.md %})<br>
