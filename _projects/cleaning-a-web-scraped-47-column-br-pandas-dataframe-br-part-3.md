---
layout: distill
title: 'Geospatial Engineering in Pandas: Creating Valid GPS Columns, Part 3'
date: 2022-01-11
description: 'Highlights my skills in geospatial data engineering using pandas. This part focuses on cleaning, validating, and creating GPS data columns by merging longitude and latitude fields into Point geometry objects.'
img: 'assets/img/838338477938@+-3948324823-data-cleansing.webp'
tags: ['geospatial-data', 'data-validation', 'GPS-data-creation', 'pandas', 'geometry-manipulation']
category: ['Data Preprocessing']
comments: true
---
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#geospatial-engineering-in-pandas-creating-valid-gps-columns-part-3">Geospatial Engineering in Pandas: Creating Valid GPS Columns, Part 3</a></div>
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

# Geospatial Engineering in Pandas: Creating Valid GPS Columns, Part 3

Highlights my skills in geospatial data engineering using pandas. This part focuses on cleaning, validating, and creating GPS data columns by merging longitude and latitude fields into Point geometry objects.

### Links To All Parts Of The Series

[Mastery in Pandas: In-Depth Data Exploration, Part 1]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-1.md %})  
[PyJanitor Proficiency: Efficient String Data Cleaning, Part 2]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-2.md %})  
[Geospatial Engineering in Pandas: Creating Valid GPS Columns, Part 3]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-3.md %})  
[Advanced Data Cleaning and Validation: Batch Processing with Pandas, Part 4]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-4.md %})  


## Summary Of The Series

> This series demonstrates my deep expertise in pandas and pyjanitor for advanced data exploration and cleaning. In Part 1, "[Mastery in Pandas: In-Depth Data Exploration, Part 1]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-1.md %})," a 47-column dataset is analyzed to showcase complex tabular data management. Dimensionality reduction techniques are applied to retain only relevant columns. Part 2, "[PyJanitor Proficiency: Efficient String Data Cleaning, Part 2]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-2.md %})," leverages pyjanitor's method chaining syntax to enhance the speed and efficiency of string data cleaning. The focus is on unique column values to guide the cleaning steps.

> Regular expressions (regex) are extensively used in Part 4, "[Advanced Data Cleaning and Validation: Batch Processing with Pandas, Part 4]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-4.md %})," for extracting and validating cell contents, replacing invalid patterns with valid ones. Additionally, this part emphasizes handling large volumes of tabular data through batch processing. Part 3, "[Geospatial Engineering in Pandas: Creating Valid GPS Columns, Part 3]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-3.md %})," highlights the creation and validation of `Point` geometry `gps` columns by merging longitude and latitude fields into GPS data columns. The series culminates with the creation of new `timedelta64[ns]` `time_listed` and `gps` columns, illustrating advanced data cleaning and validation techniques.

---

*Important Distinction:*

When we approach the process of data analysis in the following, we often selectively choose columns from our DataFrame that are most relevant to our immediate analytical goals. This means some columns may be dropped to streamline the analysis and focus on the most pertinent data. However, this step is specific to the exploratory and interpretative phases of our project.

In contrast, for the machine learning (ML) phase, our strategy differs significantly. Here, we retain a broader range of columns, including those we might have excluded during the data analysis phase. The rationale is that, even if a column doesn't seem immediately relevant for analytical insights, it could still provide valuable information when building predictive models. Therefore, we clean and prepare all columns meticulously to ensure they are suitable as input features for our ML algorithms.

This approach serves two main purposes in the ML context:

1. **Baseline Model Creation**: By including a comprehensive set of features, we establish a robust baseline model. This model serves as an initial point of reference to evaluate the performance of more complex models developed later.
    
2. **Feature Evaluation and Engineering**: With a full set of features at our disposal, we have the opportunity to explore a wide range of variables during feature engineering. This stage might reveal unexpected patterns or relationships that could enhance the predictive power of our final models.
    

In summary, the difference in handling the DataFrame between data analysis and machine learning stages is deliberate and strategic. It reflects the distinct objectives and methodologies of these two critical phases of our project.

--- 

## Heating Costs

Based on df.heating_costs.value_counts(), we can see that heating costs are
often included in auxiliary costs. There is no numerical value for 4047 rows,
which is around $$\frac{1}{3}$$ of all rows. Listings with one of the major
heating types and similar isolation should have similar heating costs (given in
$$m^{2}$$). Because Hamburg is located near the Northern Sea, its winters are
generally mild. Certainly, Hamburg's climate isn't continental, where heating
accounts for a higher percentage of rent.

Therefore, the column is removed. 


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

## Creation Of GPS Column 

**The Steps for linking listing location with location based features**. In
order to create the GPS column, the longitude and latitude columns are joined
together using geometry object `Point` from library `shapely.geometry`. This
enables completely independent geospatial features to be assigned to listings
from the dataset. These features allow for the engineering of a robust set of
features for each listing, such as proximity to the nearest underground and
suburban train station, noise levels for day and night from street noise for
example. Many prove to be significant for predicting variable 'base_rent' later
on in the process. Please see the full text of my Bachelor's Thesis
**[**Data Mining: Hyperparameter Optimization For Real Estate Prediction Models**]({% link assets/pdf/hyperparameter-optimization-bachelor-thesis-tobias-klein.pdf%})** for more details.

Furthermore, the data from the 'date_listed' is transformed into valid and
'date_unlisted' columns is calculated to determine how long a listing has been
on 'immoscout24.de'.


## Latitude

Together with the Longitude column, this column is one of the most important in
the dataset. For most listings, they provide GPS coordinates. In addition to
this spatial data, external geospatial information will be added from the
dataset that is independent of the dataset. *XGBoost* and *Lasso Regression* models
will be trained using these features together. 

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

Taking a look at a sample of the values in the Latitude column, it appears they
all follow the same pattern, and can therefore be easily converted to floating
point numbers. There is no doubt that the values from these columns are much
easier to clean than values from those from the visible features in the URL. The
`pandas.Series.str.extract()` function can be used in the following cases.  Both
columns have the same number of unique values before and after cleaning.




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
assert lngna == latna
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

We drop the `street_number`, and `floor_space` columns after the creation of
the `gps` column. `street_number` (street *and* number as part of the address
of a listing) is not needed anymore, since all listings have a valid pair of
longitude and latitude coordinates associated with them. `floor space` is a
variable that has 827 non-missing values and around 91.2% of rows have missing
values. This ratio between valid and missing values is too large to be imputed
for analytical purposes in this case.



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

Handling date-time data is a crucial step in data analysis, especially when dealing with time-sensitive information like rental listings. Our approach involves careful extraction and processing of date-time data to ensure accuracy and consistency.

1. **Data Extraction**: We begin by extracting date information from two specific columns in our dataset. The data is expected to be in the format `dd.mm.yyyy`. This step is vital to standardize the date format across our dataset, allowing for uniform processing and analysis.

2. **Utilizing Pandas `datetime64[ns]`**: Pandas offers the `datetime64[ns]` data type, which is part of the [pandas.Timestamp class](https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html#pandas-timestamp). This data type is precise up to the nanosecond. However, our data only includes information up to the day level. To align with our data granularity and improve processing efficiency, we truncate the time information, removing any details below the day level. This simplification streamlines our dataset and focuses our analysis on relevant date information.

3. **Handling Missing Data**: In cases where date information is missing, we employ a forward-fill strategy. This means we replace missing values with the next available valid entry. This approach is based on the observation that the time a listing is online usually doesn't vary significantly, with few exceptions. By using the next valid date, we maintain continuity in our data and minimize the impact of missing values. This method is particularly useful when analyzing trends over time, as it ensures no gaps in the date sequence.

 - *Note on Outliers*: While this method is generally effective, it's important to be aware of outliers. Listings that were online for unusually long or short periods can skew the analysis. We will address these outliers in a later stage, ensuring our final insights are robust and representative of the typical listing patterns.

Through these steps, we ensure that our date-time data is accurate, consistent, and ready for further analysis, such as trend identification or temporal pattern exploration.


```python
# Process the 'date_listed' and 'date_unlisted' columns to extract valid dates and convert them to datetime
df = (
    df.process_text(
        column_name="date_listed",
        string_function="extract",
        pat=r"(\d{2}\.\d{2}\.\d{4})",  # Pattern to match dates in dd.mm.yyyy format
        expand=False
    )
    .process_text(
        column_name="date_unlisted",
        string_function="extract",
        pat=r"(\d{2}\.\d{2}\.\d{4})",  # Same pattern for 'date_unlisted'
        expand=False
    )
    .to_datetime("date_listed", errors="raise", dayfirst=True)  # Convert to datetime with day first format
    .to_datetime("date_unlisted", errors="raise", dayfirst=True)  # Convert to datetime with day first format
    .fill_direction(date_listed="up", date_unlisted="up")  # Forward fill missing data
    .truncate_datetime_dataframe(datepart="day")  # Truncate time information below day level
)
```

### Exploring Data After Cleaning


```python
# Print the first 10 rows of 'date_listed' and 'date_unlisted' to inspect the cleaned data
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



## Handling Negative Time Deltas: A Critical Approach

During our data cleaning process, we've identified instances where `date_unlisted` is earlier than `date_listed`, resulting in negative time deltas. Initially, one might consider simply converting these deltas to absolute values to rectify the negatives. This approach, while straightforward, has a significant limitation: it can potentially obscure underlying data quality issues.

Negative time deltas often point towards deeper inconsistencies or errors in data entry, such as incorrect listing or unlisting dates. Merely converting these values to their absolute counterparts may yield numerically correct but contextually inaccurate data. This could lead to misleading conclusions in both our data analysis and machine learning stages.

Hence, a more nuanced and effective approach is required:

1. **Individual Investigation**: We should first scrutinize each case of negative time delta. This involves examining the corresponding `date_listed` and `date_unlisted` entries to identify potential errors or anomalies.
    
2. **Correcting Data Inconsistencies**: Wherever possible, we should attempt to correct these inconsistencies. This may involve consulting additional data sources, cross-referencing with other columns, or applying logical rules based on our understanding of the data.
    
3. **Documenting and Reporting**: All changes and the rationale behind them should be thoroughly documented. This transparency is crucial for maintaining the integrity of the dataset and for future reference.
    
4. **Fallback Strategy**: In cases where correction is not feasible, taking the absolute value of the time delta may be used as a fallback strategy. However, this should be accompanied by a note of caution regarding the potential inaccuracies it introduces.
    

By adopting this more rigorous approach, we not only enhance the reliability of our dataset but also deepen our understanding of its nuances. This leads to more robust and credible analyses and predictions in subsequent stages of our project.

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

 Given that only few rows are affected and that the inspection of the affected rows points towards swapped values for `date_listed` and `date_unlisted` for the affected listing, we keep the outlined solution.

## Creating the time_listed Column

Next, we create the time_listed column by calculating the difference between date_unlisted and date_listed. We then apply an absolute transformation to ensure all values are positive.

```python

# Add 'time_listed' column with absolute time deltas
df = (
    df.add_column(
        column_name="time_listed", 
        value=df["date_unlisted"] - df["date_listed"]
    )
    .transform_column(
        column_name="time_listed", 
        function=lambda x: np.abs(x)
    )
)

# Verify minimum time listed and check for missing values
print(df["time_listed"].min())
print(df["time_listed"].isna().value_counts())
```

    0 days 00:00:00
    False    9423
    Name: time_listed, dtype: int64

The minimum value in time_listed is correctly set to 0 days, indicating proper handling of the data.

### Dtype of `time_listed`

We confirm that the time_listed column is of type timedelta64[ns], aligning with our data processing requirements.

```python
# Check the data type of 'time_listed' column
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

Using df.describe(), we explore the distribution of time_listed values. The disparity between the mean and median suggests the presence of outliers.

```python
# Describe statistics of 'time_listed' to identify distribution and outliers
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


### Dropping Redundant Column

Finally, we remove the `pc_city_quarter` column, as it becomes redundant (for our initial analysis) due to the availability of GPS-based location data.


```python
# Drop the 'pc_city_quarter' column
df.drop(columns=["pc_city_quarter"], inplace=True)
```

By ensuring the correctness of the time delta calculations and addressing data inaccuracies, we maintain the integrity and reliability of our analysis. The above steps and code modifications enhance readability and align with best practices in Python and pandas usage.

This was part 3 in our series on data cleaning, and geospatial feature creation series. Keep reading part 4: 
[Advanced Data Cleaning and Validation: Batch Processing with Pandas, Part 4]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-4.md %})  

---
<br>
<br>

[Mastery in Pandas: In-Depth Data Exploration, Part 1]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-1.md %})  
[PyJanitor Proficiency: Efficient String Data Cleaning, Part 2]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-2.md %})  
[Geospatial Engineering in Pandas: Creating Valid GPS Columns, Part 3]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-3.md %})  
[Advanced Data Cleaning and Validation: Batch Processing with Pandas, Part 4]({% link _projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-4.md %})

---

**© Tobias Klein 2022 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
