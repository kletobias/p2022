---
layout: distill
title: 'Cleaning a web scraped 47 Column<br>Pandas DataFrame<br>Part 3'
date: 2022-01-11
description: 'Extensive cleaning and validation and creation of a valid GPS column from the records, by joining the longitude and latitude columns together using geometry object Point.'
img: 'assets/img/838338477938@+-3948324823.jpg'
tags: ['data-validation', 'dtype-timedelta64','geospatial-feature-engineering', 'pandas', 'tabular-data']
category: ['data-preprocessing']
comments: true
---
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#summary-of-this-article">Summary Of This Article</a></div>
    <div class="no-math"><a href="#summary-of-the-series">Summary Of The Series</a></div>
    <div class="no-math"><a href="#heating-costs">Heating Costs</a></div>
    <div class="no-math"><a href="#latitude">Latitude</a></div>
    <div class="no-math"><a href="#creating-the-gps-column">Creating The GPS Column</a></div>
    <div class="no-math"><a href="#creation-of-the-gps-column-13">Creation Of The GPS Column 1/3</a></div>
    <div class="no-math"><a href="#reviewing-the-columns-in-the-dataframe-again">Reviewing The Columns In The DataFrame Again</a></div>
    <div class="no-math"><a href="#date-listed--date-unlisted-columns">Date Listed & Date Unlisted Columns</a></div>
    <div class="no-math"><a href="#time_listed-column">Time_Listed Column</a></div>
  </nav>
</d-contents>

# Wrangling with that Data! 3/4

This series shows how cleaning a CSV file using `pandas`, `numpy`, `re` and the
`pyjanitor` (imported under the name `janitor`) modules can be achieved. Some
outputs are shortened for readability.

#### Links To All Parts Of The Series

[Data Preparation Series 1]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-1.md %})  
[Data Preparation Series 2]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-2.md %})  
[Data Preparation Series 3]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-3.md %})  
[Data Preparation Series 4]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-4.md %})  

## Summary Of This Article

Creation of a valid gps column for the records, by joining the
longitude and latitude columns together using geometry object `Point` from
library `shapely.geometry`. It lays the foundation for assigning from the
dataset completely independent geospatial features to the listings. Features
that prove significant for the prediction of variable 'base_rent' in the later
stages of the process. Further, a dtype `timedelta64[ns]` column is created
using `datetime64[ns]` type columns 'date_listed' and 'date_unlisted' to
calculate how long a listing was listed for on the platform
'immoscout24.de'.<br>

## Summary Of The Series

- A DataFrame is given as input that contains 47 columns at the beginning.
- Dimensionality Reduction is performed on the columns, to filter and only keep relevant columns.
- The `pyjanitor` module is widely used with its *method chaining syntax* to increase the speed of the cleaning procedure.
- Unique values of each column give the basis for the steps needed to clean the columns.
- Regular Expressions (**regex**) are mostly used to extract cell contents that hold the valid data.
- Regex are also used to replace invalid character patterns with valid ones.
- Validation of the values, after cleaning is performed using regex patterns.
- New `timedelta64[ns]` `time_listed` and `Point` geometry `gps` columns are created.


## Heating Costs

Looking at `df.heating_costs.value_counts()`, we see that the heating costs are
often included in the auxiliary costs. For 4047 rows, which is around
$$\frac{1}{3}$$ of all rows, it only states that heating costs are included in the
auxiliary without any numerical value. On average, heating costs should be
around the same for listings with one of the major heating types and similar
isolation and using normalized area inside a listing (given in $$m^{2}$$). Hamburg
is close to the Northern Sea and therefore the winters in Hamburg are generally
mild. There certainly is no continental climate, where heating make up a higher
percentage of the total rent.

The column is therefore dropped.


```python
df.heating_costs.value_counts().head(10)
```




      in Nebenkosten enthalten           4047
      nicht in Nebenkosten enthalten     1070
      keine Angabe                        731
      50 €                                248
      60 €                                220
      70 €                                206
      80 €                                199
      100 €                               199
     inkl. 50 €                           150
      90 €                                144
    Name: heating_costs, dtype: int64




```python
df.drop(labels=["heating_costs"], axis=1, inplace=True)
```

## Latitude

This column is one of the most important ones in the dataset together with the
Longitude column. Together they give the exact GPS coordinates for most of the
listings. This spacial information will be joined with from the dataset
independent external geospatial based information. Together these will lay the
foundation for the most influential features used to train the *XGBoost* and *
Lasso Regression* models.

```python
df.lat.value_counts().sample(10, random_state=seed)
```




    ['lat: 53.4859709952484,']      1
    ['lat: 53.555415063775214,']    1
    ['lat: 53.59607281949776,']     1
    ['lat: 53.575903316512345,']    1
    ['lat: 53.56514913880538,']     1
    ['lat: 53.56863155760176,']     3
    ['lat: 53.45789899804567,']     1
    ['lat: 53.60452600011545,']     1
    ['lat: 53.619085739824705,']    1
    ['lat: 53.54586549494192,']     5
    Name: lat, dtype: int64



### Cleaning Of Longitude And Latitude Columns
A look at a sample of the values found in the Latitude column, shows that they
most likely all follow the same pattern and thus can be easily converted to
floating point numbers. Generally, the `json_` columns are much easier to clean,
than the values from the visible features on the URL of the listing. For the
following cases, the `pandas.Series.str.extract()` function is the tool of
choice.  No difference in the number of unique values before and after the
cleaning for both columns is found.


```python
print(df.lat.nunique(), df.lng.nunique())
df = (
        df.process_text(
                column_name="lat",
                string_function="extract",
                pat=r"[^.]+?(\d{1,2}\.{1}\d{4,})",
                expand=False,
                )
            .process_text(
                column_name="lng",
                string_function="extract",
                pat=r"[^.]+?(\d{1,2}\.{1}\d{4,})",
                expand=False,
                )
            .change_type(column_name="lat", dtype="float64", ignore_exception=False)
            .change_type(column_name="lng", dtype="float64", ignore_exception=False)
)

print(df.lat.nunique(), df.lng.nunique())
```

    5729 5729
    5729 5729


### Validation of Longitude and Latitude Columns
Only missing values in both columns do not match the validation pattern. This
was expected, since we did not add `.dropna()` to the list comprehension over
the values in the columns.


```python
bb = [x for x in df.lat.unique().tolist() if not re.match(r"\A\d+\.\d{4,}\Z", str(x))]
ba = [x for x in df.lng.unique().tolist() if not re.match(r"\A\d+\.\d{4,}\Z", str(x))]
print(ba)
print('\n')
print(bb)
```

    [nan]


    [nan]


## Creating The GPS Column

To be able to utilize the Latitude and Longitude values that we have in the
dataset, we need to create a tuple consisting of the two variables. The result
should look like this for all valid pairs: `(lng,lat)`. In order for a tuple of
this form to be recognized as a valid GPS value, we need to be able to apply the
`.apply(Point)` method to all values and get no errors during the application.

We start by checking for problematic rows. Rows that need attention are the
following:

- Rows that only have one valid value in the subset `df[['lng','lat']]`
- Rows with missing values in both columns 'lng' and 'lat'


The output shows that all rows for the two columns are the same, in terms of
whether a row has a missing or valid value. With this information, we can save
the index values of the rows with missing values for the subset latitude and
longitude, so we can drop the rows with missing values in the gps column without
much effort in next steps.


```python
df[["lng", "lat"]].isna().value_counts()
```




    lng    lat  
    False  False    9423
    True   True     2901
    dtype: int64



The method above is to be preferred over the one below, which creates lists for
both columns `lng` and `lat`, with the index values of the rows with *NaN*
values. It then compares the index values stored in `lngna` and `latna`
elementwise. The `assert` function is used to confirm that the elements in both
lists are all equal.


```python
lngna = list(df[df["lng"].isna()].index)
latna = list(df[df["lat"].isna()].index)
assert lngna == lngna
```

## Creation Of The GPS Column 1/3
The rows of the `lng` and `lat` columns are added together using the `tuple`
function inside a `DataFrame.apply()` statement and the result is assigned to
the new column `gps`.


```python
from shapely.geometry import Point
```


```python
df["gps"] = df[["lng", "lat"]].apply(tuple, axis=1)
```

We use the row index values from earlier. This makes it easy for us to drop the
missing values, regardless of the fact that a tuple of missing values can not be
dropped by using `df.dropna(subset=['gps'],inplace=True)` anymore. Such a tuple
is not an `np.nan`, it just has values `( np.nan, np.nan )`inside the tuple.


```python
df.drop(lngna, inplace=True, axis=0)
```

### Drop Rows That Contain *NaN* Values
We drop rows that have *NaN* values in the `lng` and `lat` columns. Around 3000
rows where dropped as a result.


```python
len(df)
```




    9423



No more *NaN* values in all three columns, as expected.


```python
df[["lng", "lat", "gps"]].isna().value_counts()
```




    lng    lat    gps  
    False  False  False    9423
    dtype: int64



### Creation Of The GPS Column 2/3
Just valid values for variable longitude and latitude are left and therefore only valid values make up the data in the gps column.
We check how the values in the gps column look like, before the `POINT` conversion.


```python
df["gps"].sample(1000, random_state=seed)
```




    4555      (10.081875491322997, 53.59659576773955)
    5357      (10.117258625619199, 53.57200940640784)
    5463      (10.076499662582167, 53.60477899815401)
    4191       (9.948503601122527, 53.58407791510109)
    7238        (9.944410649308205, 53.5642271656938)
                               ...                   
    10138    (10.024189216380647, 53.588549633689425)
    3479       (9.859813703847099, 53.53408132036414)
    5558        (10.03611960198576, 53.6124394982734)
    11072      (10.009835277290465, 53.5488715679383)
    4786      (9.944531180915247, 53.577945758643175)
    Name: gps, Length: 1000, dtype: object



### Conversion Of The `gps` Column To Geometry Object 3/3

The `gps` column is ready for conversion, by applying the `Point` function to all values in the column. The result needs no further processing.


```python
df["gps"] = df["gps"].apply(Point)
```

### Getting A Quick Look At The Finished GPS Column
The `gps` column looks as expected, and we can see that its dtype is correct as well:

`Name: gps, dtype: object <class 'shapely.geometry.point.Point'>`


```python
print(df["gps"][0:10], type(df["gps"][10]))
```

    0     POINT (10.152357145948395 53.52943934146104)
    1      POINT (10.06842724545814 53.58414266878421)
    2        POINT (9.86423115803761 53.6044918821393)
    3      POINT (9.956881419437474 53.56394970119381)
    4     POINT (10.081257248271337 53.60180649336605)
    5    POINT (10.032298671585043 53.561192834080046)
    6    POINT (10.204297951920761 53.493636048600344)
    7      POINT (9.973693895665868 53.56978432240341)
    8      POINT (9.952844267614122 53.57611822321049)
    9     POINT (10.036249068267281 53.56479904983683)
    Name: gps, dtype: object <class 'shapely.geometry.point.Point'>



```python
df[['gps']].info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9423 entries, 0 to 12323
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   gps     9423 non-null   object
    dtypes: object(1)
    memory usage: 405.3+ KB


## Reviewing The Columns In The DataFrame Again

We drop more columns, after the creation of the `gps` column. `street_number`
(street and number of a listing) are not needed anymore, since all listings have
a valid pair of longitude and latitude coordinates associated with them. `floor
space` is a variable that has 827 non-missing values and around 91.2% of rows
have missing values. This ratio between valid and missing values is too large to
be imputed in this case. The column is therefore dropped.



```python
df.drop(columns=["street_number", "floor_space"], inplace=True)
```

## Date Listed & Date Unlisted Columns
For the columns that give information about when a listing was published and
unpublished on the rental platform the needed cleaning steps are identical.
Therefore, not all steps are explicitly described for both columns.  The values
need to be converted to dtype datetime, so the entire column can be assigned
dtype `datetime64[ns]`.



```python
df.date_listed.unique()[
0:10
]  # Entire list of unique values was used for the following steps.
```




    array(['[\'exposeOnlineSince: "03.12.2018"\']',
           '[\'exposeOnlineSince: "02.12.2018"\']',
           '[\'exposeOnlineSince: "30.11.2018"\']',
           '[\'exposeOnlineSince: "29.11.2018"\']',
           '[\'exposeOnlineSince: "28.11.2018"\']',
           '[\'exposeOnlineSince: "27.11.2018"\']',
           '[\'exposeOnlineSince: "26.11.2018"\']',
           '[\'exposeOnlineSince: "25.11.2018"\']',
           '[\'exposeOnlineSince: "24.11.2018"\']',
           '[\'exposeOnlineSince: "23.11.2018"\']'], dtype=object)



### Extracting DateTime Information
- For both columns, we extract the valid data. Data in the form of `dd.mm.yyyy`.
- pandas `datetime64[ns]` goes down to the Nanosecond level. However, the data only goes down to the day level. This leads to us removing anything below the day level in the data.
- We fill missing data by selecting the next valid data entry, above the row with the missing value. The time a listing was online does not vary much, except for a few outliers, as will be discussed later.


```python
df = (
        df.process_text(
                column_name="date_listed",
                string_function="extract",
                pat=r"(\d{2}\.\d{2}\.\d{4})",
                expand=False,
                )
            .process_text(
                column_name="date_unlisted",
                string_function="extract",
                pat=r"(\d{2}\.\d{2}\.\d{4})",
                expand=False,
                )
            .to_datetime("date_listed", errors="raise", dayfirst=True)
            .to_datetime("date_unlisted", errors="raise", dayfirst=True)
            .fill_direction(date_listed="up", date_unlisted="up")
            .truncate_datetime_dataframe(datepart="day")
)
```

### Exploring Data After Cleaning


```python
ppr = df[["date_listed", "date_unlisted"]][0:10]
print(ppr)
```

      date_listed date_unlisted
    0  2018-12-03    2018-12-03
    1  2018-12-03    2018-12-03
    2  2018-12-03    2018-12-03
    3  2018-12-03    2018-12-03
    4  2018-12-03    2018-12-03
    5  2018-12-02    2018-12-02
    6  2018-11-30    2018-12-03
    7  2018-11-30    2018-12-03
    8  2018-11-30    2018-12-03
    9  2018-12-03    2018-12-03


The time listed column is created calculating the difference between the
`date_unlisted` and `date_listed` columns. The result is the `time_listed`
column, which has type `timedelta64[ns]`. We truncate the timedelta values in
this column to only show the days that the listing was online, since our data
only includes the day a listing was listed/unlisted.


```python
tg = df["date_unlisted"] - df["date_listed"]
```

Looking at several metrics for the `timedelta64[ns]` type column, we see that
there are negative values for a couple of rows. Looking at the corresponding
values of the `date_listed` and `date_unlisted` columns, we see that they most
likely are in the wrong order. The negative values are dealt with, after this
inspection, by using the absolute value of all `timedelta64[ns]` values. This
only corrects the negative deltas, while not altering the positive ones.


```python
print(tg.min())
print(tg.max())
print(tg.mean())
print(tg.median())
print(tg.quantile())
print(tg[tg.dt.days < 0].index.tolist())
indv = tg[tg.dt.days < 0].index.tolist()

df.loc[indv, ["date_listed", "date_unlisted"]]
```

    -648 days +00:00:00
    924 days 00:00:00
    19 days 05:10:04.011461318
    5 days 00:00:00
    5 days 00:00:00
    [41, 578, 919, 1161, 1218, 1581, 2652, 2682, 2869, 2959, 3150, 3629, 3686, 6833, 7543, 7777, 7794, 11283, 11570, 11795, 11829, 11842, 11965, 12023]






<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date_listed</th>
      <th>date_unlisted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>2018-11-29</td>
      <td>2018-11-28</td>
    </tr>
    <tr>
      <th>578</th>
      <td>2018-10-26</td>
      <td>2018-10-25</td>
    </tr>
    <tr>
      <th>919</th>
      <td>2018-10-08</td>
      <td>2018-10-07</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>2018-09-24</td>
      <td>2018-09-23</td>
    </tr>
    <tr>
      <th>1218</th>
      <td>2018-09-20</td>
      <td>2018-09-19</td>
    </tr>
    <tr>
      <th>1581</th>
      <td>2018-11-01</td>
      <td>2018-09-03</td>
    </tr>
    <tr>
      <th>2652</th>
      <td>2018-09-21</td>
      <td>2018-09-20</td>
    </tr>
    <tr>
      <th>2682</th>
      <td>2018-09-03</td>
      <td>2018-07-08</td>
    </tr>
    <tr>
      <th>2869</th>
      <td>2018-08-08</td>
      <td>2018-06-28</td>
    </tr>
    <tr>
      <th>2959</th>
      <td>2018-08-08</td>
      <td>2018-06-22</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>2018-10-08</td>
      <td>2018-06-13</td>
    </tr>
    <tr>
      <th>3629</th>
      <td>2018-06-15</td>
      <td>2018-05-15</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>2018-07-12</td>
      <td>2018-06-04</td>
    </tr>
    <tr>
      <th>6833</th>
      <td>2018-11-23</td>
      <td>2017-12-12</td>
    </tr>
    <tr>
      <th>7543</th>
      <td>2017-08-24</td>
      <td>2017-08-23</td>
    </tr>
    <tr>
      <th>7777</th>
      <td>2017-09-21</td>
      <td>2017-09-08</td>
    </tr>
    <tr>
      <th>7794</th>
      <td>2018-01-24</td>
      <td>2017-08-08</td>
    </tr>
    <tr>
      <th>11283</th>
      <td>2018-08-17</td>
      <td>2017-01-26</td>
    </tr>
    <tr>
      <th>11570</th>
      <td>2018-09-21</td>
      <td>2016-12-12</td>
    </tr>
    <tr>
      <th>11795</th>
      <td>2018-09-10</td>
      <td>2017-07-07</td>
    </tr>
    <tr>
      <th>11829</th>
      <td>2018-09-10</td>
      <td>2017-09-22</td>
    </tr>
    <tr>
      <th>11842</th>
      <td>2018-10-01</td>
      <td>2017-07-28</td>
    </tr>
    <tr>
      <th>11965</th>
      <td>2018-06-06</td>
      <td>2017-10-30</td>
    </tr>
    <tr>
      <th>12023</th>
      <td>2018-06-21</td>
      <td>2017-03-22</td>
    </tr>
  </tbody>
</table>



## time_listed Column
The `time_listed` column is created and added to the DataFrame.


```python
df = (
        df.add_column(
                column_name="time_listed", value=df["date_unlisted"] - df["date_listed"]
                )
            #    .change_type(column_name="time_listed", dtype="pd.Timedelta", ignore_exception=False)
            .transform_column(column_name="time_listed", function=lambda x: np.abs(x))
)
```

The minimum is 0 days, as it should be and no missing data was added by the
cleaning steps.


```python
print(df["time_listed"].min())
print(df["time_listed"].isna().value_counts())
```

    0 days 00:00:00
    False    9423
    Name: time_listed, dtype: int64


### Dtype of `time_listed`
The dtype of the new column `date_listed` is `timedelta64[ns]`, as it should be.


```python
df[['time_listed']].info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9423 entries, 0 to 12323
    Data columns (total 1 columns):
     #   Column       Non-Null Count  Dtype          
    ---  ------       --------------  -----          
     0   time_listed  9423 non-null   timedelta64[ns]
    dtypes: timedelta64[ns](1)
    memory usage: 405.3 KB


### Validation Of `time_listed`
`df.describe()` is used to show the distribution of `time_listed`, to verify
that the subtraction of `date_listed` from `date_unlisted` earlier resulted in
valid `timedelta64[ns]` values in the `time_listed` column.  It becomes clear,
that there must be outliers in the data of the `time_listed` column, as the mean
is 20 days, while the median is 5 days. The mean is much more sensitive to
outliers in the way it is calculated compared to the median.


```python
df[['time_listed']].describe()
```



<table>
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>time_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9423</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>20 days 01:41:28.252148997</td>
    </tr>
    <tr>
      <th>std</th>
      <td>46 days 13:04:26.285141156</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1 days 00:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5 days 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>21 days 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>924 days 00:00:00</td>
    </tr>
  </tbody>
</table>



Column `pc_city_quarter` is dropped, since the gps column makes it redundant.


```python
df.drop(columns=["pc_city_quarter"], inplace=True)
```

---
<br>
<br>
[Data Preparation Series 1]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-1.md %})  
[Data Preparation Series 2]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-2.md %})  
[Data Preparation Series 3]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-3.md %})  
[Data Preparation Series 4]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-4.md %})  
