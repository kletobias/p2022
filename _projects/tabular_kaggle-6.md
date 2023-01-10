---
layout: distill
title: 'Deep Dive Tabular Data Pt. 6'
date: 2023-01-09
description: 'Kaggle Submission 1'
img: 'assets/img/838338477938@+-791693336.jpg'
tags: ['deep learning', 'fastai', 'pandas', 'tabular data', 'hypterparameter optimization']
category: ['deep learning']
authors: 'Tobias Klein'
comments: true
---
<br>
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

