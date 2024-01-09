---
layout: distill
title: 'Expert Analysis in Lasso Regression: The Datacamp Concrete Challenge'
date: 2022-12-26
description: 'This article documents my approach to the Datacamp Concrete Challenge, completed within an hour. It offers detailed explanations for each step, emphasizing my proficiency with the Lasso regression model in a time-constrained environment.'
img: 'assets/img/838338477938@+-791693336.jpg'
tags: ['time-efficient-analysis', 'lasso-regression-expertise', 'mathematical-insights', 'multivariate-regression', 'regression-analysis']
category: ['Tabular Data']
authors: 'Tobias Klein'
comments: true
---
<br>
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#sub-one-hour-regression-challenge">Sub One Hour Regression Challenge</a></div>
    <div class="no-math"><a href="#the-challenge">The Challenge</a></div>
    <div class="no-math"><a href="#the-data">The Data</a></div>
    <div class="no-math"><a href="#imports">Imports</a></div>
    <div class="no-math"><a href="#creating-dependent--independent-variable-splits">Creating Dependent & Independent Variable Splits</a></div>
    <div class="no-math"><a href="#create-train--test-split">Create Train & Test Split</a></div>
    <div class="no-math"><a href="#data-preprocessing">Data Preprocessing</a></div>
    <div class="no-math"><a href="#lasso-in-depth">Lasso In-Depth</a></div>
    <div class="no-math"><a href="#final-coefficients-and-intercept">Final Coefficients And Intercept</a></div>
    <div class="no-math"><a href="#answer-to-challenge-question-part-2">Answer To Challenge Question Part 2</a></div>
    <div class="no-math"><a href="#answer-to-challenge-question-part-1">Answer To Challenge Question Part 1</a></div>
  </nav>
</d-contents>


# Sub One Hour Regression Challenge

## The Challenge
Provide your project leader with a formula that estimates the compressive strength. Include:

1. The average strength of the concrete samples at 1, 7, 14, and 28 days of age.
2. The coefficients $$\beta_{0}$$, $$\beta_{1}$$ ... $$\beta_{8}$$, to use in the following formula:

$$ Concrete \ Strength = \beta_{0} \ + \ \beta_{1}*cement \ + \ \beta_{2}*slag \ + \ \beta_{3}*fly \ ash  \ + \ \beta_{4}*water \ + $$
$$ \beta_{5}*superplasticizer \ + \ \beta_{6}*coarse \ aggregate \ + \ \beta_{7}*fine \ aggregate \ + \ \beta_{8}*age $$

## The data
The team has already tested more than a thousand samples ([source](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)):

### Compressive strength data:
- "cement" - Portland cement in kg/m3
- "slag" - Blast furnace slag in kg/m3
- "fly_ash" - Fly ash in kg/m3
- "water" - Water in liters/m3
- "superplasticizer" - Superplasticizer additive in kg/m3
- "coarse_aggregate" - Coarse aggregate (gravel) in kg/m3
- "fine_aggregate" - Fine aggregate (sand) in kg/m3
- "age" - Age of the sample in days
- "strength" - Concrete compressive strength in megapascals (MPa)

>Acknowledgments: I-Cheng Yeh, "Modeling of strength of high-performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).

## Imports

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import janitor
import numpy as np

seed = 42
```


```python
df = pd.read_csv("data/concrete_data.csv")
df.head()
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
<table class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cement</th>
      <th>slag</th>
      <th>fly_ash</th>
      <th>water</th>
      <th>superplasticizer</th>
      <th>coarse_aggregate</th>
      <th>fine_aggregate</th>
      <th>age</th>
      <th>strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.986111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.887366</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.269535</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.052780</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.296075</td>
    </tr>
  </tbody>
</table>
</div>


## Creating Dependent & Independent Variable Splits


```python
indep, dep = df.get_features_targets(
    target_column_names=["strength"],
)
```

## Create Train & Test Split

We split independent variables and the dependent variable into train and test data.
By creating a function that does this for us, we make the code more reproducible
and more robust against changes in the inputs. It also makes it easier for us to
change the train/test ratio without having to add more code to do so. It works
for any `pandas.DataFrame` that has the dependent column specified as a
DataFrame.


```python
def trntst(df, dep_col, test_size):
    indep, dep = df.get_features_targets(target_column_names=[dep_col])
    indep, dep = indep.to_numpy(), dep.to_numpy()
    indep_trn, indep_tst, dep_trn, dep_tst = train_test_split(
        indep, dep, test_size=test_size, random_state=seed
    )
    dep_trn, dep_tst = dep_trn.reshape(len(dep_trn),), dep_tst.reshape(
        len(dep_tst),
    )
    return indep_trn, indep_tst, dep_trn, dep_tst
```

Use the function defined above to create four subsets of the original DataFrame.

The train/test subsets are for the independent variables:

- `indep_tr` and `indep_tst`

And for the dependent variable these are:

- `dep_trn` and `dep_tst`


```python
indep_trn, indep_tst, dep_trn, dep_tst = trntst(df, "strength", 0.2)
```

## Data Preprocessing

Using the defaults for `sklearn.KFold` (5 folds) for cross validation, and the
linear model Lasso regression from `sklearn.linear_models.LassoCV`, which
performs cross validation to find the best value for $$\ell_1$$ regularization
parameter $$\alpha$$, when fitted to data. That is, it iterates over
different values for $$\alpha$$ and subsets of the training data.

## Lasso In-Depth

Lasso is an acronym for Least Absolute Selection and Shrinkage Operator. It
was first introduced in the paper (Tibshirani 1996). It is trained with the
$$\ell_1$$-Norm as regularizer in its default configuration with $$\alpha = 1$$.
What this means is that the sum of the absolute values of the coefficients ($$w$$)
is added to the ordinary least squares term. The complete term is
to be minimised regarding $$w$$ by the model during coordinate descent. The
objective function to minimise is

$$
\begin{aligned}
 &\qquad\qquad\qquad\qquad\qquad\min_{w} {\frac{1}{2n} ||X w - y||_2^2 + \alpha ||w||_1} \\
 &\mathrm{Euclidean}\;\mathrm{Norm:}\:\;\| x \|_2 =( \sum_{i=1}^n | x_i |^2)^{1/2},\;\:\: \ell_1\text{-}\mathrm{Norm:}\;\:\| x\|_1 := \sum_{i=1}^{n}| x_i|
\end{aligned}
$$

With $$n$$ the number of observations, $$X$$ the vector of the independent
variables with coefficients vector $$w$$ and $$y$$ the dependent variable. The
model solves the minimisation problem regarding $$w$$ imposed by the added
penalty term $$\alpha||w||_1 \ge 0$$. Where $$\alpha \ge 0$$ is a constant and
$$||w||_1$$ is the $$\ell_1$$-Norm of $$w$$. The value of $$\alpha$$ can be altered
and $$\alpha = 1$$ gives the $$\ell_1$$-Norm of $$w$$ and thus tends to *shrink*
many coefficients. The applied *shrinkage* gives coefficients with values that
are either equal to zero or close to zero (sparse coefficients). The lower
bound is $$\alpha = 0$$, which eliminates the penalty term completely and gives
an ordinary least squares linear regression. Hence the usage of the model not
only for predictions, but also as a feature selector with $$\alpha > 0$$. A
special case where ordinary least squares linear regression fails and Lasso is
used instead, is the following. Consider $$X$$ as a matrix $$X \in \mathbb{R}^{n
\times p}$$ with the number of rows given by $$n$$ (number of observations) and
the number of columns given by $$p$$ (number of regressors). Lasso is frequently
used when there are far more regressors than there are observations that is
if $$p \gg n$$. As with a general regression model, the assumption is that the
observations are either independent or that the $$y_i$$s are conditionally
independent given the $$x_{ij}$$s.

## Final Coefficients And Intercept

We get the final coefficients of the model using method `coef_` on the
estimator and the intercept by `.intercept_`.



```python
kfold = KFold()
model = LassoCV()
fitted = model.fit(indep_trn, dep_trn)
coeffs = model.coef_
intercept = model.intercept_
```


We map the coefficients to the independent variables. The coefficients of the
final model are the output of the following function.


```python
def final_params(indep, coeffs, intercept):
    return dict(
        zip([col for col in indep.columns.tolist()] + ['intercept'], np.append(coeffs,intercept))
    )
```

## Answer To Challenge Question Part 2

All we have to do now is plug in the values for each of the input variables.
The output is the answer to the second part of the question posed in the
datacamp challenge.


```python
final_params(indep, coeffs, intercept)
```




    {'cement': 0.11084314870584021,
     'slag': 0.09778986094861224,
     'fly_ash': 0.07512685155415182,
     'water': -0.21209276166636368,
     'superplasticizer': 0.0,
     'coarse_aggregate': 0.0,
     'fine_aggregate': 0.009994803080481951,
     'age': 0.11255389788043503,
     'intercept': 18.996750218740445}



## Answer To Challenge Question Part 1

The first part of the question is answered using `.groupby`, a powerful pandas
data grouping method.
We only select the columns mentioned in the first part of the question that
is columns `age` and `strength`. We group by age, which automatically sorts
the `DataFrameGroupBy` object by the keys of the column we grouped by.
Conveniently, the first five rows are the durations we are supposed to compute
the average (=mean) concrete strength for. Using `.mean` on the groupby object
returns the value we are looking for.


```python
grouped = df[["age", "strength"]].groupby("age")
gm = grouped.mean()[0:5]
gm
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>strength</th>
    </tr>
    <tr>
      <th>age</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>9.452716</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.981082</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26.050623</td>
    </tr>
    <tr>
      <th>14</th>
      <td>28.751038</td>
    </tr>
    <tr>
      <th>28</th>
      <td>36.748480</td>
    </tr>
  </tbody>
</table>
</div>
