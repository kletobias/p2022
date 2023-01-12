---
layout: distill
title: 'Cleaning a 47 Column<br>Pandas DataFrame<br>Part 2'
date: 2022-01-11
description: 'More efficient string data cleaning by using the pyjanitor module and method chaining.'
img: 'assets/img/838338477938@+-3948324823.jpg'
tags: ['data cleaning', 'pandas', 'regular expressions', 'string manipulation', 'tabular data']
category: ['data preprocessing']
comments: true
---

<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#summary-of-the-series">Summary Of The Series</a></div>
    <div class="no-math"><a href="#reading-in-the-dataframe-from-previous-steps">Reading In The DataFrame From Previous Steps</a></div>
    <div class="no-math"><a href="#checking-data-types-of-the-columns">Checking Data Types Of The Columns</a></div>
    <div class="no-math"><a href="#columns-with-little-information">Columns With Little Information</a></div>
    <div class="no-math"><a href="#has-fitted-kitchen---bool">Has Fitted Kitchen - Bool</a></div>
    <div class="no-math"><a href="#has-elevator---bool">Has Elevator - Bool</a></div>
    <div class="no-math"><a href="#auxiliary-costs--total-rent">Auxiliary Costs & Total Rent</a></div>
  </nav>
</d-contents>

# Cleaning that Data! 2/4

This article shows how cleaning a CSV file using `pandas`, `numpy`, `re` and the
`pyjanitor` (imported under the name `janitor`) modules can be achieved. Some
outputs are shortened for readability.

#### Links To All Parts Of The Series

[Data Preparation Series 1]({% link _projects/data_prep_1.md %})  
[Data Preparation Series 2]({% link _projects/data_prep_2.md %})  
[Data Preparation Series 3]({% link _projects/data_prep_3.md %})  
[Data Preparation Series 4]({% link _projects/data_prep_4.md %})  

## Summary Of The Series

- A DataFrame is given as input, that contains 47 columns at the beginning.
- Dimensionality Reduction is performed on the columns, to filter and only keep relevant columns.
- The `pyjanitor` module is widely used with its *method chaining syntax* to increase the speed of the cleaning procedure.
- Unique values of each column give the basis for the steps needed to clean the columns.
- Regular Expressions (**regex**) are mostly used to extract cell contents, that hold the valid data.
- Regex are also used to replace invalid character patterns with valid ones.
- Validation of the values, after cleaning is performed using regex patterns.
- New `timedelta64[ns]` `time_listed` and `Point` geometry `gps` columns are created.



We begin by importing the needed libraries and modules.


```python
import re  # Python built-in regular expression library.
import numpy as np
import pandas as pd
import janitor

seed = 42  # Random seed, to give repeatable random output.
```

## Reading In The DataFrame From Previous Steps

A value we alter, compared to its default value is the following:

`na_values=['np.nan','[]']`

In addition to the new label (`np.nan`) for missing values, that we applied at
the end of the previous step we can see, that '[]' marks missing values in any
of the columns, with a 'json\_' prefix. By marking '[]' values as *NaN* values
we can skip a lot of column by column reassignments later for missing values,
not recognized as such. The '[]', marking missing values, comes from the design
of the web scraping algorithm, that appended an empty list, if there was no
value for a variable.

The names of most of the columns are changed, in order for column names to be
English. Names of columns with boolean values are altered to mark these columns
with only boolean values. Columns without a `json_` prefix, that have a `json_`
counterpart have their names aligned with the one of their counterpart.


```python
df = (
        pd.read_csv(
                "/Volumes/data/bachelor_thesis/df_concat.csv",
                na_values=["np.nan", "[]"],
                index_col=False,
                )
            .rename_column("einbau_kue", "bfitted_kitchen")
            .rename_column("lift", "belevator")
            .rename_column("nebenkosten", "auxiliary_costs")
            .rename_column("gesamt_miete", "total_rent")
            .rename_column("heiz_kosten", "heating_costs")
            .rename_column("str", "street-number")
            .rename_column("nutzf", "floor_space")
            .rename_column("regio", "pc_city_quarter")
            .rename_column("online_since", "date_listed")
            .rename_column("baujahr", "yoc")
            .rename_column("objekt_zustand", "object_condition")
            .rename_column("heizungsart", "heating_type")
            .rename_column("wesent_energietr", "main_es")
            .rename_column("endenergiebedarf", "total_energy_need")
            .rename_column("kaltmiete", "base_rent")
            .rename_column("quadratmeter", "square_meters")
            .rename_column("anzahl_zimmer", "no_rooms")
            .rename_column("balkon/terasse", "bbalcony")
            .rename_column("keller", "cellar")
            .rename_column("typ", "type")
            .rename_column("etage", "floor")
            .rename_column("anz_schlafzimmer", "no_bedrooms")
            .rename_column("anz_badezimmer", "no_bathrooms")
            .rename_column("haustiere", "bpets_allowed")
            .rename_column("nicht_mehr_verfg_seit", "date_unlisted")
            .rename_column("json_heatingType", "json_heating_type")
            .rename_column("json_totalRent", "json_total_rent")
            .rename_column("json_yearConstructed", "json_yoc")
            .rename_column("json_firingTypes", "json_main_es")
            .rename_column("json_hasKitchen", "json_bfitted_kitchen")
            .rename_column("json_yearConstructedRange", "json_const_time")
            .rename_column("json_baseRent", "json_base_rent")
            .rename_column("json_livingSpace", "json_square_meters")
            .rename_column("json_condition", "json_object_condition")
            .rename_column("json_petsAllowed", "json_bpets_allowed")
            .rename_column("json_lift", "json_belevator")
            .clean_names()  # make all column names lower-case, replace space with underscore.
            .remove_empty()  # Drop rows, that only have missing values.
)
```

## Checking Data Types Of The Columns

The output shows us, that only 2 columns are of type numeric. After the cleaning
process, the columns will all have the correct **data type (dtype)**. The dtype,
of the cleaned values in each column. All columns, that have a json prefix and a
counterpart amongst the non json columns, are likely to hold the same data as
their counterparts. The columns with a json prefix were mainly scraped to
validate the data in their counterparts. The goal is to be efficient in
comparing `json_`, non `json_` column values.

Some columns will not have their 'correct' dtypes assigned to them in this
article. This comes from the fact, that to assign certain dtypes there must not
be any missing values present for all rows of the column. We do not impute any
missing values here, nor do we drop a large amount of rows, to get a DataFrame
without any missing values. Except for the Longitude (`lng`) and Latitude(`lng`)
columns, as they are some of the most important columns in the dataset.

The reason being, that we have no knowledge in regard to the 'correct'
replacement value for any of the missing values. Which of the techniques is used
to impute missing data, is tied to the model choice and the results that a model
delivers, given the applied imputation method.

This applies to the decision of dropping a certain amount of rows, in order to
remove missing values in the key columns as well. That is, the ones that are
most important for the model performance. Therefore, imputation and the decision
whether to drop rows and or columns must be evaluated in the greater context of
the predictive modeling problem at hand.

We get the number of columns in the dataset.


```python
len(df.columns)
```




    47



Column names are dtypes of columns are checked. Column names should all be lower-case and should not contain any spaces. The names all look fine.


```python
df.data_description
```




                                  type  count  pct_missing description
    column_name                                                       
    bfitted_kitchen             object   8024     0.348913            
    belevator                   object   2975     0.758601            
    auxiliary_costs             object  12324     0.000000            
    total_rent                  object  12324     0.000000            
    heating_costs               object  12324     0.000000            
    lat                         object   9423     0.235394            
    lng                         object   9423     0.235394            
    street_number               object   9482     0.230607            
    floor_space                 object   1185     0.903846            
    pc_city_quarter             object  12324     0.000000            
    parking                     object   3187     0.741399            
    date_listed                 object  12324     0.000000            
    yoc                         object  11342     0.079682            
    object_condition            object   7810     0.366277            
    heating_type                object   9526     0.227037            
    main_es                     object   9930     0.194255            
    total_energy_need           object   3771     0.694012            
    base_rent                   object  12324     0.000000            
    square_meters               object  12324     0.000000            
    no_rooms                    object  12324     0.000000            
    bbalcony                    object   8526     0.308179            
    cellar                      object   6337     0.485800            
    type                        object   9703     0.212674            
    floor                       object   9737     0.209916            
    no_bedrooms                float64   6469     0.475089            
    no_bathrooms               float64   7317     0.406280            
    bpets_allowed               object   3914     0.682408            
    date_unlisted               object  11629     0.056394            
    json_heating_type           object   9968     0.191172            
    json_balcony                object  12324     0.000000            
    json_electricitybaseprice   object  12321     0.000243            
    json_picturecount           object  12324     0.000000            
    json_telekomdownloadspeed   object  11626     0.056637            
    json_telekomuploadspeed     object  11626     0.056637            
    json_total_rent             object  11220     0.089581            
    json_yoc                    object  11137     0.096316            
    json_electricitykwhprice    object  12321     0.000243            
    json_main_es                object  11504     0.066537            
    json_bfitted_kitchen        object  12324     0.000000            
    json_cellar                 object  12324     0.000000            
    json_const_time             object  11137     0.096316            
    json_base_rent              object  12324     0.000000            
    json_square_meters          object  11762     0.045602            
    json_object_condition       object  12324     0.000000            
    json_interiorqual           object  12324     0.000000            
    json_bpets_allowed          object  12324     0.000000            
    json_belevator              object  12324     0.000000            



## Columns With Little Information

We get an overview of the number of unique values for each column in the
DataFrame.


```python
va = []
for key, val in np.ndenumerate(df.columns.tolist()):
    va.append(df[val].nunique())
va = pd.Series(data=va, index=df.columns)
```

The 5 columns with the fewest unique values are all boolean dtype columns. This
was to be expected, however it does not mean that little information is found in
these columns.


```python
print(f"The columns with the 5 smallest nunique values are:\n{va.nsmallest(5)}")
```

    The columns with the 5 smallest nunique values are:
    bfitted_kitchen    1
    belevator          1
    bbalcony           1
    cellar             1
    json_balcony       2
    dtype: int64


Boolean type columns are marked by prepending character `b` to the column name,
after any `json_` substring in the column name. The values for boolean type
columns are expected to have 2 unique values, once cleaned. It has to be
checked, which boolean columns have valid values, 2 unique values. These have
certain characteristics, as shown below.

- One value that marks the absence of the variable the column represents, e.g. 0, 'n', 'no'.
- The second value, that shows, that the variable is present for a given record or row in the dataset. E.g. 1, 'y', 'yes'.
- Both will be converted to a numerical value, if the dtype differs from numerical (`int64`).

Columns, that are dropped, looking at the output from the following cells, are:  

- `json_electricitybaseprice` & `json_electricitykwhprice`  

Both columns only have 3 unique values (not including any missing values) and
one value is found in all but 183 of the 12324 listings. That is an electricity
base price of 90.76 Eur. for 12138 listing, leaving 183 that have a lower base
price of 71.43 Eur.

- The `json_telekomdownloadspeed` & `json_telekomuploadspeed`  

Show the available internet speeds from provider Telekom. The download and
upload speeds of internet provider Telekom are the same for around $\frac{2}{3}$
of all listings. Upload and download speeds have the same distribution and so
likely have a perfect positive correlation with each other, making one
redundant. `json_telekomdownloadspeed` is kept for further evaluation.


```python
print(
        f"Sorted nunique values, in ascending order:\n\n{va.sort_values(axis=0, ascending=True)}"
        )
```

    Sorted nunique values, in ascending order:

    bfitted_kitchen                 1
    belevator                       1
    cellar                          1
    bbalcony                        1
    json_cellar                     2
    json_bfitted_kitchen            2
    json_electricitybaseprice       2
    json_balcony                    2
    json_belevator                  2
    json_electricitykwhprice        3
    bpets_allowed                   3
    json_bpets_allowed              4
    json_interiorqual               5
    no_bathrooms                    5
    json_telekomuploadspeed         6
    json_telekomdownloadspeed       6
    no_bedrooms                     8
    object_condition                9
    json_const_time                 9
    json_object_condition          10
    type                           10
    heating_type                   12
    json_heating_type              13
    json_main_es                   14
    no_rooms                       19
    parking                        32
    main_es                        37
    json_picturecount              47
    floor                         120
    json_yoc                      155
    yoc                           156
    floor_space                   349
    date_unlisted                 701
    pc_city_quarter               752
    date_listed                   801
    total_energy_need            1019
    heating_costs                1530
    auxiliary_costs              2352
    json_square_meters           2725
    square_meters                3136
    base_rent                    4112
    json_base_rent               4112
    json_total_rent              4369
    total_rent                   5191
    street_number                5724
    lat                          5729
    lng                          5729
    dtype: int64



```python
fv = [
        "json_electricitybaseprice",
        "json_electricitykwhprice",
        "json_telekomdownloadspeed",
        "json_telekomuploadspeed",
        ]
for key in fv:
    print(f"\n{df[key].value_counts()}")
```


    ['"obj_electricityBasePrice":"90.76"']    12138
    ['"obj_electricityBasePrice":"71.43"']      183
    Name: json_electricitybaseprice, dtype: int64

    ['"obj_electricityKwhPrice":"0.1985"']    12137
    ['"obj_electricityKwhPrice":"0.2205"']      183
    ['"obj_electricityKwhPrice":"0.2195"']        1
    Name: json_electricitykwhprice, dtype: int64

    ['"obj_telekomDownloadSpeed":"100 MBit/s"']    8208
    ['"obj_telekomDownloadSpeed":"50 MBit/s"']     2756
    ['"obj_telekomDownloadSpeed":"16 MBit/s"']      605
    ['"obj_telekomDownloadSpeed":"25 MBit/s"']       48
    ['"obj_telekomDownloadSpeed":"200 MBit/s"']       5
    ['"obj_telekomDownloadSpeed":"6 MBit/s"']         4
    Name: json_telekomdownloadspeed, dtype: int64

    ['"obj_telekomUploadSpeed":"40 MBit/s"']     8208
    ['"obj_telekomUploadSpeed":"10 MBit/s"']     2756
    ['"obj_telekomUploadSpeed":"2,4 MBit/s"']     600
    ['"obj_telekomUploadSpeed":"5 MBit/s"']        48
    ['"obj_telekomUploadSpeed":"1 MBit/s"']         9
    ['"obj_telekomUploadSpeed":"100 MBit/s"']       5
    Name: json_telekomuploadspeed, dtype: int64



```python
df.drop(columns=[col for col in fv if col != "json_telekomdownloadspeed"], inplace=True)
```

## Has Fitted Kitchen - Bool

The value, generated from scraping the visible listing might suggest, that about
4000 missing values are found in this column. However, one can argue, that there
is reason to look at it differently. The `json_bfitted_kitchen` values show, how
the company the that runs the immoscout24 portal thinks of it. The json data is
not visible to the visitor, it summarizes the listings values and holds many
more that are not visible to the visitor.

They have assigned the boolean value of 'no' (`n`) to the missing values found
in the `bfitted_kitchen` column. If one substitutes the missing values with the
values in the `json_bfitted_kitchen` column, that have the same row index, the
columns are the same.  Another reason not to drop the column is that a listing
with a fitted kitchen will usually be a value adding feature, that increases the
market value of the listing compared to listings without a fitted kitchen. The
reason being, that a listing without a fitted kitchen will typically mean, that
the tenant will have to buy a kitchen for the apartment. The costs of buying a
kitchen are high, plus the fact, that a kitchen is often custom-made to fit the
space of the specific kitchen and thus can not be moved to another apartment.

This gives enough reason to fill the rows, that have missing values in
`bfitted_kitchen` with '0' - does not have a fitted kitchen.

### Comparing Values of `bfitted_kitchen` and `json_bfitted_kitchen`

The show the same value count for listing has a fitted kitchen. The number of
*NaN* values in column `bfitted_kitchen` is equivalent to the value count of
value `['"obj_hasKitchen":"n"']` in the `json_bfitted_kitchen` column. We use
the values of `json_bfitted_kitchen` to fill the missing values in column
`bfitted_kitchen`. We will see, that all no row in `bfitted_kitchen` has an
*NaN* value anymore.


```python
print(
        f"value_counts for bfitted_kitchen:\n{df['bfitted_kitchen'].value_counts()},\n\n json_bfitted_kitchen:\n{df['json_bfitted_kitchen'].value_counts()}"
        )
```

    value_counts for bfitted_kitchen:
    Einbauküche    8024
    Name: bfitted_kitchen, dtype: int64,

     json_bfitted_kitchen:
    ['"obj_hasKitchen":"y"']    8024
    ['"obj_hasKitchen":"n"']    4300
    Name: json_bfitted_kitchen, dtype: int64




## Has Elevator - Bool

With the same reasoning, that led to the decision to not drop the `bfitted_kitchen` column, the missing values in the elevator colum are replaced with 0.


```python
df = (
    df
    .fill_empty(
            column_names=["bfitted_kitchen", "belevator"], value=0
            )
    .find_replace(
        match="exact",
        bfitted_kitchen={
                "Einbauküche": 1
                },
        belevator={
                "Personenaufzug": 1
                },
        )
)

print(df.bfitted_kitchen.value_counts(normalize=True))
print(df.belevator.value_counts(normalize=True))
```

    1    0.651087
    0    0.348913
    Name: bfitted_kitchen, dtype: float64
    0    0.758601
    1    0.241399
    Name: belevator, dtype: float64


## Auxiliary Costs & Total Rent

We focus on how to efficiently clean and validate the values in the
`auxiliary_costs` column. There are several problems in this column, as the
output below shows. The `total_rent` column needs similar cleaning steps as
`auxiliary_costs` and so both a processed together in the following.


```python
[s for s in df.auxiliary_costs if re.match("[^\d,.]+", str(s))][0:10]
```




    ['  190,84  ',
     '  117,95  ',
     '  249,83  ',
     '  70  ',
     '  213,33  ',
     '  150  ',
     '  145  ',
     '  250  ',
     '  100  ',
     '  50  ']



There are 2352 rows in the `auxiliary_costs` column, that have character classes
other than:
- digit: [0-9] or [\d] comma: [,] or ',' period: [.] or '\\.'

These 3 characters are the only ones that should be present in the colum for
non-missing values.


```python
bb = [u for u in df.auxiliary_costs.unique() if re.match(r"[^\d,.]", str(u))]
print(len(bb))
```

    2352


No recognized *NaN* values in the column. We create a copy of the
`auxiliary_costs` column, before cleaning its values. The reason for this, is
that *NaN* values showed after the cleaning step.


```python
no_na_auxil_df = df[["auxiliary_costs"]]

```


```python
lov = df["auxiliary_costs"][df.auxiliary_costs.isna()]
lov
```




    Series([], Name: auxiliary_costs, dtype: object)



No string values, that represent missing values, but label 'keine Angabe', which
is equivalent to *NaN*. This only became obvious after the cleaning steps, since
all alphabetic characters were dropped during the cleaning, leaving rows with
'keine Angabe' entries empty. Pandas in turn assigned *NaN* values to these
rows.


```python
lov = df['auxiliary_costs'][df['auxiliary_costs'].str.contains(pat=r"[a-zA-Z\\s]+") == True][0:10]
lov
```




    160       keine Angabe
    657       keine Angabe
    766       keine Angabe
    788       keine Angabe
    905       keine Angabe
    1027      keine Angabe
    1188      keine Angabe
    1250      keine Angabe
    1407      keine Angabe
    1414      keine Angabe
    Name: auxiliary_costs, dtype: object



No rows, that don't have at least one numerical value.


```python
ltv = df['total_rent'][df['total_rent'].str.contains(pat=r"\A[^\d]\Z") == True]
ltv
```




    Series([], Name: total_rent, dtype: object)



There are no recognized missing values in column `total_rent` prior to cleaning.


```python
df.total_rent.isna().value_counts()
```




    False    12324
    Name: total_rent, dtype: int64



### Unique Values
The unique values are the basis upon which any cleaning is performed. If all
problems found in the unique values of any column are addressed, then the column
is considered clean. Missing values are a unique value, but require filling
methods or the rows containing them need to be dropped. There is no universal
solution, when it comes to dealing with missing values.<br>
<br>
For most other problems where more than a reassignment of the dtype of the
values in the column is needed, regular expressions are used to create patterns
to surgically remove the problems, while preserving the valid parts of the
data.<br>
After using regular expressions to extract, substitute, remove or reorder parts
of the cell content, and with the correct substitutions where needed, values in
that specific column are in the format they should be in. At this point, the
correct dtype can be assigned to all rows in the column without raising any
errors during the
reassignment.<br>
The reassignment might not be possible without raising errors, if
missing values are present. In this case, the missing values need to be
addressed, prior to reassigning the dtypes.


```python
for i in ["auxiliary_costs", "total_rent"]:
    j = df[i].unique()
    print(f"\nUnique values in {i}:\n\n{j}")
```


    Unique values in auxiliary_costs:

    ['  190,84  ' '  117,95  ' '  249,83  ' ... '  89,59  ' '  197,96  '
     '  59,93  ']

    Unique values in total_rent:

    [' 541,06  ' ' 559,05  ' ' 839,01  ' ... ' 1.189,24  ' ' 382,73  '
     ' 499,91  ']


The cleaning steps turned all non-missing values into floating point numbers
with no errors being raised during the conversion. We still want to validate,
that the values in both columns only contain digits and optionally a `.` as
decimal separator, followed by nothing else than 2 digits at the end of each
value.

### Detailed Cleaning Steps For Total Rent And Auxiliary Costs

Next up is the actual cleaning procedure for columns `auxiliary_costs` and
`total_rent`. The problem with these variables is, that it is unknown, which of
the following optional variables, the ones inside `[]` are factored in at all or
to what degree:

$$\mathrm{total\,\, rent} = \mathrm{base \,\,rent} + \mathrm{auxiliary \,\,costs} + [\mathrm{heating \,\,costs} + \mathrm{X}]$$

$$\mathrm{X}$$ stands for several costs, that might or might not be factored in.
These variables will likely not make it to the machine learning stage, since
there is a high chance, that they are correlated with the dependent variable
`base_rent`. For now, we shall simply be efficient in cleaning them and further
structured exploration will tell how to proceed with these two variables.

<br>

#### Cleaning Steps
The cleaning steps are illustrated by the example of `total_rent`. The steps
apply to the `auxiliary_costs` column as well, see the code below.  We start by
looking at its value counts, to get an idea of what regex pattern are needed to
transform it into a column, that has dtype `float`. The rows, that need most
cleaning are ones, with entries like this: `1.050  (zzgl. Heizkosten)`. Things
that need attention are:
- Drop any spaces, be it `\s` or `\t` in any number, anywhere.
- Get rid of `.` as a thousand separator, while substituting `,` with `.` as a decimal separator. Worst case for numerical values looks like this  `4.514,49`.
- Drop any parenthesis and whatever is inside them, if it can not be the total rent value inside the parenthesis.
- Convert the dtype of the column to `float`, without any errors during the conversion.
The conversion to float, gave no errors.
We check for missing values after the conversion and drop 1 row of the DataFrame, where `total_rent` has the missing value.


```python
df = (
        df
            .process_text(
                column_name="auxiliary_costs", string_function="lstrip", to_strip=r"[^\d]+"
                )
            .process_text(
                column_name="auxiliary_costs", string_function="rstrip", to_strip=r"[^\d]+"
                )
            .process_text(
                column_name="total_rent", string_function="lstrip", to_strip=r"[^\d]+"
                )
            .process_text(
                column_name="total_rent", string_function="rstrip", to_strip=r"[^\d]+"
                )
            .process_text(
                column_name="auxiliary_costs",
                string_function="extract",
                pat=r"([\d,.]+)",
                expand=False,
                )
            .process_text(
                column_name="total_rent",
                string_function="extract",
                pat=r"([\d,.]+)",
                expand=False,
                )
           .process_text(
                column_name="auxiliary_costs",
                string_function="replace",
                pat=r"(\d{1,2})\.(\d{3})",
                repl=r"\1\2",
                )
            .process_text(
                column_name="total_rent",
                string_function="replace",
                pat=r"(\d{1,2})\.(\d{3})",
                repl=r"\1\2",
                )
            .process_text(
                column_name="auxiliary_costs", string_function="replace", pat=",", repl="."
                )
            .process_text(
                column_name="total_rent", string_function="replace", pat=",", repl="."
                )
            .change_type(
                column_name="auxiliary_costs", dtype="float64", ignore_exception=False
                )
            .change_type(
                column_name="total_rent", dtype="float64", ignore_exception=False
                )
)
```



### Dealing With Newly Created Missing Values
The cleaning process introduced around 200 missing values in the
`auxiliary_costs` column, that were not recognized as such by *pandas*, prior to
cleaning.


```python
print(df.auxiliary_costs.isna().value_counts())
```

    False    12121
    True       203
    Name: auxiliary_costs, dtype: int64


We test, what value these new *NaN* values had before the cleaning step, by
extracting the index given by `df['auxiliary_costs].isna()`. This index contains
the numbers in this set: $$\{0,1\}$$.
- Value of $$0\longrightarrow$$ row-index with no missing value.
- Value of $$1\longrightarrow$$ row-index with missing value.


```python
y = df['auxiliary_costs'].isna().tolist()
```

We get confirmation, that there were missing values in the `auxiliary_costs`
column before cleaning. We only found these by going through the list of unique
values in that column earlier. After the removal of any alphabetic characters in
the column, value 'keine Angabe' was replaced with *NaN* by pandas. As mentioned
earlier, we do not drop the rows with missing date for now.


```python
no_na_auxil_df.loc[y]
```



<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>auxiliary_costs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>160</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>657</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>766</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>788</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>905</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>12112</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>12120</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>12164</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>12171</th>
      <td>keine Angabe</td>
    </tr>
    <tr>
      <th>12174</th>
      <td>keine Angabe</td>
    </tr>
  </tbody>
</table>
<p>203 rows × 1 columns</p>



Still no *NaN* values in column `total_rent` after cleaning.


```python
print(df.total_rent.isna().value_counts())
```

    False    12324
    Name: total_rent, dtype: int64


### Total Rent Post Cleaning Validation
The conversion of `total_rent` to float gave no errors. The column is converted
into a list and all values pass the validation. That means they only contain
digits before a period (`.`), followed by exactly 2 digits after the period
before the end of the string. Regex syntax for checking that only the end of
string follows what is matched by the pattern is `\Z`, in the case of the `re`
module: `r'some_pattern\Z'`. Similarly for making a pattern start matching
whatever comes at the very beginning of a string, using the `re` module, `\A` is
used , like so: `r'\Asome_pattern'`.


```python
bb = [
        x
        for x in df.total_rent.unique().tolist()
        if not re.match(r"\A\d+\.\d{1,2}\Z", str(x))
        ]
print(bb)
```

    []


We validate the non-missing data in `auxiliary_costs` with a regex pattern.

Validation of values is done by checking the format of all entries in the
`df['auxiliary_costs']` column, that are not `np.nan` values.  All rows with
valid data pass the validation.


```python
print(
        sum(
                [
                        cc
                        for cc in pd.Series(df["auxiliary_costs"].dropna())
                        if not re.match(r"\A\d+[.]?\d+?\Z", str(cc))
                        ]
                )
        )  # no matches
ll = [
        cc
        for cc in pd.Series(df["auxiliary_costs"].dropna())
        if not re.match(r"\A\d+[.]?\d*?\Z", str(cc))
        ]  # no matches
print(ll)
```

    0
    []

---
<br>
<br>
[Data Preparation Series 1]({% link _projects/data_prep_1.md %})  
[Data Preparation Series 2]({% link _projects/data_prep_2.md %})  
[Data Preparation Series 3]({% link _projects/data_prep_3.md %})  
[Data Preparation Series 4]({% link _projects/data_prep_4.md %})  
