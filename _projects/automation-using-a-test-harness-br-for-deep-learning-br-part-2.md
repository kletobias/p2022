---
layout: distill
title: 'Automation Using A Test Harness<br>For Deep Learning:<br>Part 2'
date: 2022-12-15
description: 'This is Part 2 in the series, where we explore how the fastai deep learning library can be used to conduct structured empirical experiments on a novel and small dataset. The dataset consists of 850 images and an almost uniform distribution for the target labels. There are two labels in total, "male" and "female" that are assigned the gender of the model depicted in any of the images in the dataset.'
img: 'assets/img/838338477938@+-67822330.jpg'
tags: ['binary-classification', 'deep-learning', 'fastai', 'hyperparameter-optimization','image-data']
category: ['deep-learning']
authors: 'Tobias Klein'
comments: true
---

<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#batch-no-1">Batch No. 1</a></div>
    <div class="no-math"><a href="#importing-the-csv-file">Importing The CSV File</a></div>
    <div class="no-math"><a href="#initial-look-at-the-dataframe">Initial Look At The DataFrame</a></div>
    <div class="no-math"><a href="#dropping-columns-that-are-not-needed">Dropping Columns That Are Not Needed</a></div>
    <div class="no-math"><a href="#dropping-rows-that-are-not-needed">Dropping Rows That Are Not Needed</a></div>
    <div class="no-math"><a href="#observations-from-looking-at-the-dataframe">Observations From Looking At The DataFrame</a></div>
    <div class="no-math"><a href="#grouping-the-data">Grouping The Data</a></div>
    <div class="no-math"><a href="#summary-batch-no-1">Summary: Batch No. 1</a></div>
    <div class="no-math"><a href="#batch-no-2">Batch No. 2</a></div>
    <div class="no-math"><a href="#csv-to-dataframe">CSV To DataFrame</a></div>
    <div class="no-math"><a href="#creation-of-the-dataframe-for-analysis">Creation Of The DataFrame For Analysis</a></div>
    <div class="no-math"><a href="#grouped-by-split_seed-and-model">Grouped By split_seed And model</a></div>
    <div class="no-math"><a href="#the-worst-accuracy-993-percent">The Worst Accuracy: 99.3 Percent</a></div>
    <div class="no-math"><a href="#summary-batch-no-1--2">Summary: Batch No. 1 & 2</a></div>
  </nav>
</d-contents>

# Part 2: Analyzing The Results

This is *Part 2* in the series, where we explore how the fastai deep learning
library can be used to conduct structured empirical experiments on a novel and
small dataset. The dataset consists of 850 images and an almost uniform
distribution for the target labels. There are two labels in total, 'male'
and 'female' that are assigned the gender of the model depicted in any of the
images in the dataset.

## Batch No. 1


This article is the sequel to *Batch No. 1*. Here we look at the data logged during the testing of the first
batch in the series. Everything leading up to where we start in this article
is found in the first article [**Part 1: Basic Automation For Deep Learning**]({% link
_projects/automation-using-a-test-harness-br-for-deep-learning-br-part-1.md %}).

### The Imports

We will need pandas to work with the tabular data that is stored in
a CSV file. Pandas is needed for most of the analyzing done. One
can find information on the commands used in the following, by looking at the
pandas docs: [API reference — pandas 1.4.3 documentation](https://pandas.pydata.org/docs/reference/index.html)

The `pyjanitor` library (imported as `janitor`) adds quality of life
improvements in the form of convenient wrappers for common pandas functions and
methods. These are mainly used for cleaning tabular data stored in a
`pandas.DataFrame` or `pandas.Series`.

The pyjanitor library does so, by using *method chaining*, inspired by the `R`
package called *janitor*. Follow the link, for more information, including the
docs of this library: [pyjanitor documentation](https://pyjanitor-devs.github.io/pyjanitor/)

From matplotlib, we import pyplot. Pyplot is a general tool for plotting and
visualizing data in Python. The docs can be found here:
[API Reference — Matplotlib 3.5.2 documentation](https://matplotlib.org/stable/api/index)

Various parts of the `fastai` library are used throughout the following. One can
find its docs following this link: [fastai - Welcome to fastai](https://docs.fast.ai/)


```python
import itertools
import fastai
import fastai.vision.models
from fastai.vision.all import *
import fastcore
from fastai.test_utils import *
from pathlib import Path
import numpy as np
import re
import ipywidgets
import pandas as pd
import janitor
import matplotlib.pyplot as plt
```

## Importing The CSV File

The first thing to do, is to import the DataFrame that holds the results from
the first series of experiments that we conducted in the first article. Batch
No. 1 is how we will refer to them in the following.



```python
df = pd.read_csv("batch1-df.csv")
print(df.columns)
```

    Index(['Unnamed: 0', 'unique_setup', 'model', 'fine_tune', 'valid_pct',
           'train_loss', 'valid_loss', 'error_rate', 'lr'],
          dtype='object')


## Initial Look At The DataFrame

Looking at the output of the `df.columns` command for the data from the first
batch, we can see that there is one column named `Unnamed: 0`. This column is
always the first one when importing any CSV file that was exported to CSV using
`pandas.DataFrame.to_csv` without specifying `index=False`.

```python
df
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
<caption><strong>Table 1</strong> Column <i>unique_setup</i> gives the unique
setups whether there are one or two epochs of <i>fine_tune</i> for each of the
setups. Further details for every setup, as well as performance metrics can be
found in the corresponding columns.</caption>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>unique_setup</th>
      <th>model</th>
      <th>fine_tune</th>
      <th>valid_pct</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>lr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0-0</td>
      <td>resnet34</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.068911</td>
      <td>0.000718</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1-0</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.085328</td>
      <td>0.006074</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1-1</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.046834</td>
      <td>0.000495</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2-0</td>
      <td>resnet34</td>
      <td>1</td>
      <td>0.4</td>
      <td>0.073126</td>
      <td>0.003532</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3-0</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.049529</td>
      <td>0.002324</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>3-1</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.047148</td>
      <td>0.001684</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>4-0</td>
      <td>resnet18</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.041904</td>
      <td>0.002066</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>5-0</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.067504</td>
      <td>0.031245</td>
      <td>0.011765</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>5-1</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.058283</td>
      <td>0.014664</td>
      <td>0.011765</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>6-0</td>
      <td>resnet18</td>
      <td>1</td>
      <td>0.4</td>
      <td>0.060523</td>
      <td>0.006704</td>
      <td>0.005882</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>7-0</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.037862</td>
      <td>0.009111</td>
      <td>0.005882</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>7-1</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.078054</td>
      <td>0.008371</td>
      <td>0.005882</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>


## Dropping Columns That Are Not Needed


The `Unnamed: 0` column is dropped, as discussed and along with
it, columns `unique_setup`, `lr`, and `train_loss` as well.

Unique setup is no longer needed, since we know, which rows belong to one epoch
setups (these have fine_tune=1) and, which to two epoch setups (fine_tune=2).
This can be found looking at the column `fine_tune`.

The learning rate parameter was not touched during the experiments and so all
values for learning rate (`lr`) are the default value of 0.001.

`train_loss` is not needed, since we are only interested in the performance of
the model on the validation set, not on the training set. The loss on the
validation set after each epoch relative to the error rate on the validation set
is what we are interested in.


```python
df = df.remove_columns(column_names=["Unnamed: 0", "unique_setup", "lr", "train_loss"])
```

The result is a leaner version of the initial `df`. There is one more step to
complete thought, for the DataFrame to be ready for analysis.

## Dropping Rows That Are Not Needed


As can be seen by looking at the index of the DataFrame (very first column with
integer values from 0 to 11), there are a total of 12 rows in the DataFrame
right now. However, there were only 8 different setups that were created in
total. The additional 4 rows come from the setups that use a `fine_tune` value
of 2. The first epoch of all setups with *fine_tune == 2* is of no interest to
us, and so we only keep the second epoch for these setups. The results of the
first epoch for these setups can not be compared to the first epoch of any setup
that uses *fine_tune == 1*, since it is the final epoch for the latter, but not
for the one using *fine_tune==2*. Therefore, only the second occurrence of value
2 is kept for all rows that have value 2 in two consecutive rows.

This concludes the initial cleaning of the DataFrame. Please see the output
below, for the lines of code used to clean the DataFrame.


```python
rows = [0, 2, 3, 5, 6, 8, 9, 11]
df = df.iloc[rows].reset_index(drop=True)
df
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
<caption><strong>Table 2</strong> A slimmer version of Table 1.</caption>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>fine_tune</th>
      <th>valid_pct</th>
      <th>valid_loss</th>
      <th>error_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>resnet34</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.000718</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>resnet34</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.000495</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>resnet34</td>
      <td>1</td>
      <td>0.4</td>
      <td>0.003532</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>resnet34</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.001684</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>resnet18</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.002066</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>resnet18</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.014664</td>
      <td>0.011765</td>
    </tr>
    <tr>
      <th>6</th>
      <td>resnet18</td>
      <td>1</td>
      <td>0.4</td>
      <td>0.006704</td>
      <td>0.005882</td>
    </tr>
    <tr>
      <th>7</th>
      <td>resnet18</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.008371</td>
      <td>0.005882</td>
    </tr>
  </tbody>
</table>
</div>


## Observations From Looking At The DataFrame

It becomes obvious by looking at the DataFrame that for 3 out of the 4 tested
setups that use *resnet18* as model, the error_rate (on the
validation set) is larger than 0. The worst recorded error rate is given for the
following configuration in the output below. We will come to the error rate
for the *resnet34* configurations in a little.

```python
df.filter_on('error_rate > 0.008', complement=False)
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
      <th>model</th>
      <th>fine_tune</th>
      <th>valid_pct</th>
      <th>valid_loss</th>
      <th>error_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>resnet18</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.014664</td>
      <td>0.011765</td>
    </tr>
  </tbody>
</table>
</div>


## Grouping The Data

With the DataFrame only having the essential columns now, the analysis can start. Pandas offers a method that can be
applied to any `pandas.DataFrame` and `pandas.Series` object for analysis.

The first group call splits the data by distinct values of the `model` column. It returns the minimum value grouped by
corresponding unique values for each of the other columns in the DataFrame. The only columns of interest, given this
groupby call, are the `error_rate` and the `valid_loss`.


It should be kept in mind that the input values used here come from a small sample, both in number of images in total
and model combinations assessed. Essentially, we look for correlations in the data, which would require further testing.
Nonetheless, analyzing and looking for patterns, as done here, in the results is always possible.


```python
gb = df.groupby(by="model")
gb.median()
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
      <th>fine_tune</th>
      <th>valid_pct</th>
      <th>valid_loss</th>
      <th>error_rate</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>resnet18</th>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.007538</td>
      <td>0.005882</td>
    </tr>
    <tr>
      <th>resnet34</th>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.001201</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


Interestingly, the median (to understand what it is and its difference to the
average/mean metric: [Median - Wikipedia](https://en.wikipedia.org/wiki/Median))
of `valid_loss` for both models is within the same order of magnitude. The `resnet34`
still has the edge over the `resnet18` when it comes to the `valid_loss`.

Another interesting observation however is that the median of the `error_rate`
on the validation dataset is 0. That means that there is no setup in which the
*resnet34* was used, with an error rate other than 0. That is very intriguing,
since it is the deeper model of the two and suggests that the added depth,
equivalent to the added layers in this instance is beneficial to the model's
performance.

Let us look at the *resnet34* in more detail then.


```python
gb = (
    df.filter_string(column_name="model", search_string="34")
    .groupby("valid_pct")
    .value_counts()
)
gb
```




    valid_pct  model     fine_tune  valid_loss  error_rate
    0.2        resnet34  1          0.000718    0.0           1
                         2          0.000495    0.0           1
    0.4        resnet34  1          0.003532    0.0           1
                         2          0.001684    0.0           1
    dtype: int64



The output shows that a validation percentage of 40 percent caused a higher
loss on the validation set, compared to the lower and default 20 percent value.
The difference is one order of magnitude. That is the only difference between
valid_pct of 0.2 and 0.4.

Another observation worth mentioning is that a `fine_tune` value of two as
opposed to one, gives a slightly lower loss on the validation set. The
difference however is within the same order of magnitude and therefore this
finding might not be confirmed when conducting further empirical experiments.

This is about as much, as the small initial experiments on my *teenager models
dataset* can deliver in understanding how the different setups affect the final
results.

## Summary: Batch No. 1

### The Good

Overall, the results of most of the combinations tested are almost *too good to be true.*
With most having a final error rate of 0 on the validation set.

### The Bad

I say too good to be true, in hindsight of the possibility that all the tested
models might have severe problems to predict the target label, when being shown
an out of sample image set of the two models. Images where the models are
photographed in many other poses, scenes, lighting, just to name a few that come
to mind.

### The Solution

In general, we try to get an idea of how the models' performance will be on
unseen data, try to challenge it by using cross validation techniques.

While there is no solution that can rule out all of these uncertainties, there
is something that can be tested given this dataset.

Test different seeds for the `RandomSplitter` parameter, used in the initialization
of the `DataBlock` object. A seed of 42 was used throughout the experiments,
summarized in this article as (batch 1).
A different seed could lead to a different split, which in turn could give a different
sample of images used for training and testing.

Adhering to the principle of *only change one parameter at a time*, when
conducting structured empirical experiments, only parameter `split_seed` was
changed during the testing of *batch no. 2* and its results analyzed.

## Batch No. 2

The following links lead to the documentation pages of the most important
objects and callbacks used throughout this article. They are all part of the
fastai deep learning library. 

**DataBlock**
[fastai - Data block tutorial](https://docs.fast.ai/50_tutorial.datablock.html#building-a-datablock-from-scratch)

**dataloader**
[https://docs.fast.ai/data.load.html#dataloader](https://docs.fast.ai/data.load.html#dataloader)

**vision_learner**
[fastai - Vision learner](https://docs.fast.ai/vision.learner.html#vision_learner)

**fine_tune**
[fastai - Hyperparam schedule](https://docs.fast.ai/callback.schedule.html#learner.fine_tune)


For batch 2, we only need two of the libraries that we imported for batch 1
earlier. Please see the beginning of this article for descriptions of the
imported libraries.

```python
import pandas as pd
import janitor
```

## CSV To DataFrame

We load the DataFrame, using the alias `df` and drop the columns that we don't
need in the following (see batch 1 for more details).


```python
df = pd.read_csv("csv/df-batch2.csv")

df = df.remove_columns(column_names=["Unnamed: 0", "lr", "train_loss", "valid_loss"])
df
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
        <th>setup</th>
        <th>epochs</th>
        <th>model</th>
        <th>fine_tune</th>
        <th>valid_pct</th>
        <th>error_rate</th>
        <th>split_seed</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>0</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>8</td>
      </tr>
      <tr>
        <th>1</th>
        <td>1</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>23</td>
      </tr>
      <tr>
        <th>2</th>
        <td>2</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>42</td>
      </tr>
      <tr>
        <th>3</th>
        <td>3</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>7</td>
      </tr>
      <tr>
        <th>4</th>
        <td>4</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>8</td>
      </tr>
      <tr>
        <th>5</th>
        <td>5</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>23</td>
      </tr>
      <tr>
        <th>6</th>
        <td>6</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>42</td>
      </tr>
      <tr>
        <th>7</th>
        <td>7</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>7</td>
      </tr>
      <tr>
        <th>8</th>
        <td>8</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>8</td>
      </tr>
      <tr>
        <th>9</th>
        <td>9</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>23</td>
      </tr>
      <tr>
        <th>10</th>
        <td>10</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>42</td>
      </tr>
      <tr>
        <th>11</th>
        <td>11</td>
        <td>1</td>
        <td>resnet34</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>7</td>
      </tr>
      <tr>
        <th>12</th>
        <td>12</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>8</td>
      </tr>
      <tr>
        <th>13</th>
        <td>13</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.005882</td>
        <td>23</td>
      </tr>
      <tr>
        <th>14</th>
        <td>14</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>42</td>
      </tr>
      <tr>
        <th>15</th>
        <td>15</td>
        <td>2</td>
        <td>resnet34</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>7</td>
      </tr>
      <tr>
        <th>16</th>
        <td>16</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>8</td>
      </tr>
      <tr>
        <th>17</th>
        <td>17</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>23</td>
      </tr>
      <tr>
        <th>18</th>
        <td>18</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.005882</td>
        <td>42</td>
      </tr>
      <tr>
        <th>19</th>
        <td>19</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>7</td>
      </tr>
      <tr>
        <th>20</th>
        <td>20</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>8</td>
      </tr>
      <tr>
        <th>21</th>
        <td>21</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.000000</td>
        <td>23</td>
      </tr>
      <tr>
        <th>22</th>
        <td>22</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.005882</td>
        <td>42</td>
      </tr>
      <tr>
        <th>23</th>
        <td>23</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.2</td>
        <td>0.011765</td>
        <td>7</td>
      </tr>
      <tr>
        <th>24</th>
        <td>24</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.005882</td>
        <td>8</td>
      </tr>
      <tr>
        <th>25</th>
        <td>25</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>23</td>
      </tr>
      <tr>
        <th>26</th>
        <td>26</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.011765</td>
        <td>42</td>
      </tr>
      <tr>
        <th>27</th>
        <td>27</td>
        <td>1</td>
        <td>resnet18</td>
        <td>1</td>
        <td>0.4</td>
        <td>0.005882</td>
        <td>7</td>
      </tr>
      <tr>
        <th>28</th>
        <td>28</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>8</td>
      </tr>
      <tr>
        <th>29</th>
        <td>29</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>23</td>
      </tr>
      <tr>
        <th>30</th>
        <td>30</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.005882</td>
        <td>42</td>
      </tr>
      <tr>
        <th>31</th>
        <td>31</td>
        <td>2</td>
        <td>resnet18</td>
        <td>2</td>
        <td>0.4</td>
        <td>0.000000</td>
        <td>7</td>
      </tr>
    </tbody>
  </table>
  </div>



## Creation Of The DataFrame For Analysis

The steps taken to create the DataFrame used to analyze the results are, in this
order:

1. Only use a subset of columns: `df[["split_seed","model","error_rate"]]`. The
   output is a DataFrame again.

2. Group the remaining 'error_rate' column by 'split_seed' and 'model'.

3. Aggregate 'error_column' by median, mean and standard deviation.

4. Finally, sort the resulting DataFrame by the values in the
   `('error_rate','mean')` column (multi-index) in ascending order.


```python
gb = (
    df[["split_seed", "model", "error_rate"]]
    .groupby(by=["split_seed", "model"])
    .agg(["median", "mean", "std"])
    .sort_values(by=("error_rate", "mean"))
)
gb
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead tr th {
      text-align: left;
  }

  .dataframe thead tr:last-of-type th {
      text-align: right;
  }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">error_rate</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>median</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>split_seed</th>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <th>resnet34</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <th>resnet34</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <th>resnet18</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <th>resnet34</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <th>resnet18</th>
      <td>0.000000</td>
      <td>0.001471</td>
      <td>0.002941</td>
    </tr>
    <tr>
      <th>23</th>
      <th>resnet34</th>
      <td>0.000000</td>
      <td>0.001471</td>
      <td>0.002941</td>
    </tr>
    <tr>
      <th>7</th>
      <th>resnet18</th>
      <td>0.002941</td>
      <td>0.004412</td>
      <td>0.005632</td>
    </tr>
    <tr>
      <th>42</th>
      <th>resnet18</th>
      <td>0.005882</td>
      <td>0.007353</td>
      <td>0.002941</td>
    </tr>
  </tbody>
</table>
</div>



## Grouped By split_seed And model

Splitting by columns `split_seed` and `model` gives an array with 8 rows. We
only care about the median, mean, minimum and maximum value for column
`error_rate`, since this is the final metric, and the most important one
overall.

Overall, the values in the array show that the deeper *resnet34* model has a
lower median and mean value in three out of the four logged cases. The only
occurrence where this is not true, is the setup that uses `split_seed = 23`.
This could hint at the train/test split being an important element for the
models' performance on unseen data. It could negatively affect the models'
performance on out of sample data. This concern was mentioned earlier already
and given these results, remains relevant. It is not possible to quantify the
uncertainty surrounding the model's performance on unseen data, data that is not
part of this data set (neither in the train, nor the test split). Overall, both
models perform extremely well, regardless of the values for split_seed.

Testing even more values for split_seed would be something that could be tested
in the pursuit of us generating enough data, to be able to use probabilistic
reasoning, to quantify the uncertainty using probabilistic measures. E.g., *re
sampling*, *kernel density estimation*, *Bayesian probability*.


## The Worst Accuracy: 99.3 Percent

This binary image classification problem showed that both models, both
pretrained variants of the *ResNet* architecture are capable of at least nearly
flawless and often even completely flawless performance on the test set.

The 34 layer deep variant outclassed its 18 layer deep brother in three out of
the four cases. It was the `split_seed` that made the difference in the single
instance, where the 18 layer *ResNet* version edged out the 34 layer deep
version.

To put it into perspective, we compute the accuracy for all rows as fractions of
all predictions made on the test set, sorted as above in the output of *groupby*
are:


```python
gb[("error_rate", "mean")].apply(lambda x: 1 - x)
```




    split_seed  model   
    7           resnet34    1.000000
    8           resnet34    1.000000
    23          resnet18    1.000000
    42          resnet34    1.000000
    8           resnet18    0.998529
    23          resnet34    0.998529
    7           resnet18    0.995588
    42          resnet18    0.992647
    Name: (error_rate, mean), dtype: float64



It shows that even the worst performing model has an accuracy greater 99% on
the test data.

The performance of the models was so good that there was not much that could
be optimized. Nonetheless, the testing showed that the choice of how to split
the data into train and test split could be an issue, when the model has to
predict out of sample images. Solutions to try, in order to be able quantify the
likelihood of either of the two models performing in a certain way on unseen
data was discussed at the end of section: **Grouped By `split_seed` And
`model`**


## Summary: Batch No. 1 & 2

In *Batch No. 1* we covered everything, from loading the images into a
`DataBlock` object, to the creation of a `dataloaders` object and then
initializing a `vision_learner` object that is ready for the transfer learning
process. The transfer learning was done by using method `fine_tune`.

All unique parameter combinations were calculated and each one was saved as a
tuple.

At this point, the focus was on creating two test harnesses:

- *input_harness*

The first harness has dictionary keys for each parameter that is tested during
the structured empirical experiments. The values for each parameter that is
tested are given as elements of a list for each key.

- *output_harness*

The output harness logged the test results for each setup and was converted to a
tidy DataFrame at the end.

The results were analyzed using the *pandas* library.

With the insights from analyzing the results of *Batch No. 1*, the process was
repeated once more during part 2: *Batch No. 2*. More values for the parameter
of interest (`split_seed`) were added and included in the output_harness and
analyzed.

Thank you very much for reading this article. Please feel free to link to this
article or write a comment in the comments section below.
