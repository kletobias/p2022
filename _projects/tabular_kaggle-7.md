---
layout: distill
title: 'Deep Dive Tabular Data Pt. 7'
date: 2023-01-07
description: 'Kaggle Submission 2: tabular\_learner deep learning estimator optimized using manual hyperparameter optimization. XGBRegressor using RandomizedSearchCV and sampling from continuous parameter distributions.'
img: 'assets/img/838338477938@+-791693336.jpg'
tags: ['hyperparameter-optimization', 'random-search', 'tabular-data', 'tabular_learner', 'xgboost-regressor']
category: ['tabular data']
authors: 'Tobias Klein'
comments: true
---
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#tabular_learner-optimization">tabular_learner Optimization</a></div>
    <div class="no-math"><a href="#xgbregressor-optimization">XGBRegressor Optimization</a></div>
    <div class="no-math"><a href="#three-model-ensemble">Three Model Ensemble</a></div>
    <div class="no-math"><a href="#kaggle-submission">Kaggle Submission</a></div>
  </nav>
</d-contents>

# Series: Kaggle Competition - Deep Dive Tabular Data
<br>
[**Deep Dive Tabular Data Part 1**]({% link _projects/tabular_kaggle-1.md %})<br>
[**Deep Dive Tabular Data Part 2**]({% link _projects/tabular_kaggle-2.md %})<br>
[**Deep Dive Tabular Data Part 3**]({% link _projects/tabular_kaggle-3.md %})<br>
[**Deep Dive Tabular Data Part 4**]({% link _projects/tabular_kaggle-4.md %})<br>
[**Deep Dive Tabular Data Part 5**]({% link _projects/tabular_kaggle-5.md %})<br>
[**Deep Dive Tabular Data Part 6**]({% link _projects/tabular_kaggle-6.md %})<br>
[**Deep Dive Tabular Data Part 7**]({% link _projects/tabular_kaggle-7.md %})<br>
<br>

# Part 7: Optimization Routines & Final Submission

## tabular_learner Optimization

With the dataloaders objects created for training, as well as for the final
predictions the `tabular_learner` has to be optimized using a manual
hyperparameter optimization routine. Given, that this dataset is relatively
small with less than 2000 rows, a 10 core CPU machine is enough for the
optimizations applied here.

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





    0.126166



## XGBRegressor Optimization

The `XGBRegressor` is a top contender when it comes to a tree ensemble based
model, that can deliver predictions of the highest accuracy, having being used
in many competition winning submissions on Kaggle and other platforms in the
past.

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
aspects from it.

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
the evaluation metric. However, the number of models can still become extremely
high. Consider $$15$$ hyperparameters and for illustration purposes assume each
one has $$40$$ possible values. If a $$5$$ fold cross validation is used to
evaluate the models, the total number of models to build is given by $$15 \cdot
40 \cdot 5 = 3000$$. To put it into perspective, with an estimated time of
$$0.68$$ seconds that it takes to build one model on the here used machine, to
build all $$3000$$ models would take $$3000 \cdot 0.68 = 34
\;\mathrm{minutes}$$. While more computational power in the form of better
hardware is a solution up to a certain point, using one of the following methods
can be comparably more efficient. Considering these weaknesses of the grid
search procedure, there are alternatives available and given the number and
value range for each parameter included in the hyperparameter optimization, a
random search is chosen over a grid search.

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


## Three Model Ensemble
An ensemble consisting of `tabular_learner`, `RandomForestRegressor` and
`XGBRegressor` is tested using equal weights for each one. The results beat the
previous ones and this ensemble is used in the final submission.


```python
ens_preds2 = (to_np(preds.squeeze()) + rf_preds + random_proba) / 3
r_mse(ens_preds2, y_val)
```




    0.124315



## Kaggle Submission

Function `val_pred` takes a dataloader and the three estimators optimized in the
previous sections as inputs and 

Since the rmse of the predictions is lowest when only using `XGBRegressor` and
`RandomForestRegressor`, only these two are used for the final predictions
submitted to Kaggle.

The exponential function is applied to the predictions of each estimator before
adding them together and dividing them by three. This re-transforms them. In
this form the predictions can be added to a DataFrame under a new column
`SalePrice` and exported as CSV file.

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
    

Entire Series:<br>
<br>
[**Deep Dive Tabular Data Part 1**]({% link _projects/tabular_kaggle-1.md %})<br>
[**Deep Dive Tabular Data Part 2**]({% link _projects/tabular_kaggle-2.md %})<br>
[**Deep Dive Tabular Data Part 3**]({% link _projects/tabular_kaggle-3.md %})<br>
[**Deep Dive Tabular Data Part 4**]({% link _projects/tabular_kaggle-4.md %})<br>
[**Deep Dive Tabular Data Part 5**]({% link _projects/tabular_kaggle-5.md %})<br>
[**Deep Dive Tabular Data Part 6**]({% link _projects/tabular_kaggle-6.md %})<br>
[**Deep Dive Tabular Data Part 7**]({% link _projects/tabular_kaggle-7.md %})<br>
<br>
