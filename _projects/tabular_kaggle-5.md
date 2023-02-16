---
layout: distill
title: 'Deep Dive Tabular Data Pt. 5'
date: 2023-01-05
description: 'Out-of-domain problem: What it is, why it is important, how to spot it and how to deal with it.'
img: 'assets/img/838338477938@+-791693336.jpg'
tags: ['feature-importance', 'model-accuracy', 'out-of-domain-problem', 'random-forest', 'tabular-data']
category: ['tabular-data']
authors: 'Tobias Klein'
comments: true
---
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#identifying-out-of-domain-data">Identifying Out-Of-Domain Data</a></div>
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

## Identifying *Out-Of-Domain Data*

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
