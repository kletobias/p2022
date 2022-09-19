---
layout: distill
title: 'Cleaning a 47 Column<br>Pandas DataFrame<br>Part 1'
description: 'Data Preparation Series: Exploring Tabular Data With pandas: An Overview Of Available Tools In The pandas Library'
img: 'assets/img/838338477938@+-3948324823.jpg'
importance: 3
tags: ['deep learning', 'tabular data', 'pandas', 'data exploration', 'introduction']
category: ['data preprocessing']
comments: true

---
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#reading-in-the-input-data">Reading In The Input Data</a></div>
    <div class="no-math"><a href="#first-look-at-the-dataframe">First Look At The DataFrame</a></div>
  </nav>
</d-contents>

# From Webscraping Data<br>To Tidy Pandas DataFrame

### Cleaning The Data! Series.<br>Part 1/4

#### Links To All Parts Of The Series

[Data Preparation Series 1]({% link _projects/data_prep_1.md %})  
[Data Preparation Series 2]({% link _projects/data_prep_2.md %})  
[Data Preparation Series 3]({% link _projects/data_prep_3.md %})  
[Data Preparation Series 4]({% link _projects/data_prep_4.md %})  

## Reading In The Input Data

The input data is split into 3 csv files, that together capture all rental
listings that were online and within the boundaries of the city of Hamburg at
the time of scraping the data. The source was a large German rental listing site
called 'Immoscout24'. [ImmoScout24 -
https://www.immobilienscout24.de](https://www.immobilienscout24.de/) is their
official brand name and URL.

Various features were extracted from the listings through the use of webscraping
and it is the main objective at this stage to clean and construct a tidy
DataFrame, that is ready for the following stages. A brief overview of the
following stages is given below.

- Feature Engineering - Adding location based features.
- EDA - Exploratory Data Analysis.
- Machine Learning - Fitting and optimizing candidate models to select the best model for this problem. Predictions are made for variable 'base rent'.
- Presenting the Solution to Stakeholders.
<br>
Back to the task at hand, we begin by reading in the csv files and creating the
Pandas (*pd*) DataFrame object.  Throughout this article, any Pandas DataFrame
object will be assigned to a variable that always contains the letters ' df',
plus any prefix or suffix, preceding or succeeding the letters 'df' in some
cases.  <br> In the first step, we import the necessary modules

```python
import pandas as pd # The library used to manipulate and to create a tidy DataFrame object
seed = 42 # Create reproducible random output.
```
<br>
The path to the input data is assigned to variables `scraping_{1..3}`. For each
of them a DataFrame object is created afterwards. The DataFrame `df`, which
holds the data of all three is created and duplicate rows are dropped.  The
command used to drop any possibly duplicate rows is `df.drop_duplicates()`
without any specifying further parameters as to the subset of columns to
consider when determining, if two rows are identical. Like that, only such rows
are dropped, that have identical values for all variables found in the dataset.
This was mainly done to get rid of overlapping page ranges from the scraping
part and also to get rid of duplicate listings on the website.

```python
scraping_1 = "../data/20181203-first_scraping_topage187.csv"
scraping_2 = "../data/20181203-second_scraping_topage340.csv"
scraping_3 = "../data/20181203-third_scraping_topage340.csv"

df1 = pd.read_csv(scraping_1, index_col=False, parse_dates=False)
df2 = pd.read_csv(scraping_2, index_col=False, parse_dates=False)
df3 = pd.read_csv(scraping_3, index_col=False, parse_dates=False)

df = df1.append([df2, df3], ignore_index=True)
del df["Unnamed: 0"]
df = df.drop_duplicates()
```

## First Look At The DataFrame

To get a first look at the newly created DataFrame `df`, one can choose between
multiple tools in the pandas library. It is assumed, that the Dataframe is named
`df` in the following, as a couple of the tools, the pandas library has to
offer, are described and links to the documentation page of each command are
added for more detail on how each command works.

- `df.head()`
  - [pandas.DataFrame.head — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html)
- The counterpart of `df.head()` is `df.tail()`
  - [pandas.DataFrame.tail — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html)
- `df.columns`
  - [pandas.DataFrame.columns — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html)
- `df.index`
  - [pandas.DataFrame.index — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.index.html)
- `df.describe()`
  - [pandas.DataFrame.describe — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
- `df.shape`
  - [pandas.DataFrame.shape — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html)
- `df.count()`
  - [pandas.DataFrame.count — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.count.html)
- `df.nunique()`
  - [pandas.DataFrame.nunique — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html)
- `df.value_counts()`
  - [pandas.DataFrame.value_counts — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html)
- `df.filter()`
  - [pandas.DataFrame.filter — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html)
- `df.sample()`
  - [pandas.DataFrame.sample — pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html)

### Commands

#### `df.head()`

##### Description<br>
The command `df.head()`returns the first 5 rows of the DataFrame by default, if
no parameters are specified by the user. Using the parameter *n*, one can
specify the number of rows, that get returned. Needless to say, rows returned
will always start at the first index value and include the following *n-1* rows.

##### Example<br>
In the first call to `df.head()`, the default value for number of lines printed
(*n*=5) is used by not specifying any parameter value in the function call. In
the second call, the number of lines printed is changed to *n* =9.

```python
df.head() # Includes index values [0:4], which is default (n=5)
```

<div style="width:656px;overflow-x:scroll;">
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>einbau_kue</th>
      <th>lift</th>
      <th>nebenkosten</th>
      <th>gesamt_miete</th>
      <th>heiz_kosten</th>
      <th>lat</th>
      <th>lng</th>
      <th>str</th>
      <th>nutzf</th>
      <th>regio</th>
      <th>...</th>
      <th>json_firingTypes</th>
      <th>json_hasKitchen</th>
      <th>json_cellar</th>
      <th>json_yearConstructedRange</th>
      <th>json_baseRent</th>
      <th>json_livingSpace</th>
      <th>json_condition</th>
      <th>json_interiorQual</th>
      <th>json_petsAllowed</th>
      <th>json_lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Einbauküche</td>
      <td>NaN</td>
      <td>190,84</td>
      <td>541,06</td>
      <td>inkl. 78 €</td>
      <td>['lat: 53.52943934146104,']</td>
      <td>['lng: 10.152357145948395']</td>
      <td>Strietkoppel 20</td>
      <td>NaN</td>
      <td>22115 Hamburg Billstedt</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"3"']</td>
      <td>['"obj_baseRent":"350.22"']</td>
      <td>['"obj_livingSpace":"76"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>117,95</td>
      <td>559,05</td>
      <td>inkl. 52,07 €</td>
      <td>['lat: 53.58414266878421,']</td>
      <td>['lng: 10.06842724545814']</td>
      <td>Naumannplatz 2</td>
      <td>NaN</td>
      <td>22049 Hamburg Dulsberg</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"n"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"1"']</td>
      <td>['"obj_baseRent":"441.1"']</td>
      <td>['"obj_livingSpace":"60"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>249,83</td>
      <td>839,01</td>
      <td>inkl. 110,58 €</td>
      <td>['lat: 53.6044918821393,']</td>
      <td>['lng: 9.86423115803761']</td>
      <td>Warthestr. 52a</td>
      <td>NaN</td>
      <td>22547 Hamburg Lurup</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"n"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"5"']</td>
      <td>['"obj_baseRent":"589.18"']</td>
      <td>['"obj_livingSpace":"75"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Einbauküche</td>
      <td>NaN</td>
      <td>70</td>
      <td>665  (zzgl. Heizkosten)</td>
      <td>nicht in Nebenkosten enthalten</td>
      <td>['lat: 53.56394970119381,']</td>
      <td>['lng: 9.956881419437474']</td>
      <td>Oelkersallee 53</td>
      <td>NaN</td>
      <td>22769 Hamburg Altona-Nord</td>
      <td>...</td>
      <td>['"obj_firingTypes":"no_information"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"2"']</td>
      <td>['"obj_baseRent":"595"']</td>
      <td>['"obj_livingSpace":"46"']</td>
      <td>['"obj_condition":"well_kept"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Einbauküche</td>
      <td>NaN</td>
      <td>213,33</td>
      <td>651,81</td>
      <td>inkl. 57,78 €</td>
      <td>['lat: 53.60180649336605,']</td>
      <td>['lng: 10.081257248271337']</td>
      <td>Haldesdorfer Str. 119a</td>
      <td>NaN</td>
      <td>22179 Hamburg Bramfeld</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"6"']</td>
      <td>['"obj_baseRent":"438.48"']</td>
      <td>['"obj_livingSpace":"52"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>


```python
df.head(n=3) # Includes index values [0:2]
```




<div style="width:656px;overflow-x:scroll;">
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>einbau_kue</th>
      <th>lift</th>
      <th>nebenkosten</th>
      <th>gesamt_miete</th>
      <th>heiz_kosten</th>
      <th>lat</th>
      <th>lng</th>
      <th>str</th>
      <th>nutzf</th>
      <th>regio</th>
      <th>...</th>
      <th>json_firingTypes</th>
      <th>json_hasKitchen</th>
      <th>json_cellar</th>
      <th>json_yearConstructedRange</th>
      <th>json_baseRent</th>
      <th>json_livingSpace</th>
      <th>json_condition</th>
      <th>json_interiorQual</th>
      <th>json_petsAllowed</th>
      <th>json_lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Einbauküche</td>
      <td>NaN</td>
      <td>190,84</td>
      <td>541,06</td>
      <td>inkl. 78 €</td>
      <td>['lat: 53.52943934146104,']</td>
      <td>['lng: 10.152357145948395']</td>
      <td>Strietkoppel 20</td>
      <td>NaN</td>
      <td>22115 Hamburg Billstedt</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"3"']</td>
      <td>['"obj_baseRent":"350.22"']</td>
      <td>['"obj_livingSpace":"76"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>117,95</td>
      <td>559,05</td>
      <td>inkl. 52,07 €</td>
      <td>['lat: 53.58414266878421,']</td>
      <td>['lng: 10.06842724545814']</td>
      <td>Naumannplatz 2</td>
      <td>NaN</td>
      <td>22049 Hamburg Dulsberg</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"n"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"1"']</td>
      <td>['"obj_baseRent":"441.1"']</td>
      <td>['"obj_livingSpace":"60"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>249,83</td>
      <td>839,01</td>
      <td>inkl. 110,58 €</td>
      <td>['lat: 53.6044918821393,']</td>
      <td>['lng: 9.86423115803761']</td>
      <td>Warthestr. 52a</td>
      <td>NaN</td>
      <td>22547 Hamburg Lurup</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"n"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"5"']</td>
      <td>['"obj_baseRent":"589.18"']</td>
      <td>['"obj_livingSpace":"75"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 47 columns</p>
</div>



#### `df.tail()`

##### Description<br>
The command `df.tail()` is the counterpart to `df.head()`, it returns the last 5
rows of the DataFrame by default, if no parameters are specified by the user.
Using the parameter *n*, one can specify the number of rows, that get returned.
Needless to say, rows returned will always end with the row at the last index
value and include the preceding *n-1* rows.

##### Example<br>
First the maximum of the index of `df` is checked, to show that the last printed
row is indeed the last value in the index of the DataFrame, other than that the
examples mirror the two from the `df.head()` command, to display their
similarities.

```python
print('The maximum value of the range index of df is %s' % df.index.max())
df.tail()
```

    The maximum value of the range index of df is 12494

<div style="width:656px;overflow-x:scroll;">
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>einbau_kue</th>
      <th>lift</th>
      <th>nebenkosten</th>
      <th>gesamt_miete</th>
      <th>heiz_kosten</th>
      <th>lat</th>
      <th>lng</th>
      <th>str</th>
      <th>nutzf</th>
      <th>regio</th>
      <th>...</th>
      <th>json_firingTypes</th>
      <th>json_hasKitchen</th>
      <th>json_cellar</th>
      <th>json_yearConstructedRange</th>
      <th>json_baseRent</th>
      <th>json_livingSpace</th>
      <th>json_condition</th>
      <th>json_interiorQual</th>
      <th>json_petsAllowed</th>
      <th>json_lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12490</th>
      <td>Einbauküche</td>
      <td>NaN</td>
      <td>80</td>
      <td>1.058</td>
      <td>47 €</td>
      <td>['lat: 53.586938610218276,']</td>
      <td>['lng: 10.016258235901141']</td>
      <td>Goldbekufer 29</td>
      <td>8 m²</td>
      <td>22303 Hamburg Winterhude</td>
      <td>...</td>
      <td>['"obj_firingTypes":"oil"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"y"']</td>
      <td>['"obj_yearConstructedRange":"2"']</td>
      <td>['"obj_baseRent":"931"']</td>
      <td>['"obj_livingSpace":"66.5"']</td>
      <td>['"obj_condition":"mint_condition"']</td>
      <td>['"obj_interiorQual":"sophisticated"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>12491</th>
      <td>Einbauküche</td>
      <td>NaN</td>
      <td>61</td>
      <td>674</td>
      <td>26 €</td>
      <td>['lat: 53.55758333359278,']</td>
      <td>['lng: 10.03986901076397']</td>
      <td>Burgstraße 34</td>
      <td>NaN</td>
      <td>20535 Hamburg Hamm-Nord</td>
      <td>...</td>
      <td>['"obj_firingTypes":"oil"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"y"']</td>
      <td>['"obj_yearConstructedRange":"2"']</td>
      <td>['"obj_baseRent":"587"']</td>
      <td>['"obj_livingSpace":"51"']</td>
      <td>['"obj_condition":"refurbished"']</td>
      <td>['"obj_interiorQual":"sophisticated"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>12492</th>
      <td>Einbauküche</td>
      <td>NaN</td>
      <td>76</td>
      <td>752</td>
      <td>70 €</td>
      <td>['lat: 53.6486450531966,']</td>
      <td>['lng: 10.039966464612842']</td>
      <td>Bei der Ziegelei 4</td>
      <td>8 m²</td>
      <td>22339 Hamburg Hummelsbüttel</td>
      <td>...</td>
      <td>['"obj_firingTypes":"oil"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"y"']</td>
      <td>['"obj_yearConstructedRange":"2"']</td>
      <td>['"obj_baseRent":"606"']</td>
      <td>['"obj_livingSpace":"63.69"']</td>
      <td>['"obj_condition":"refurbished"']</td>
      <td>['"obj_interiorQual":"sophisticated"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>12493</th>
      <td>Einbauküche</td>
      <td>Personenaufzug</td>
      <td>59,93</td>
      <td>382,73</td>
      <td>21,62 €</td>
      <td>['lat: 53.54886472789119,']</td>
      <td>['lng: 10.079737639604678']</td>
      <td>Culinstraße 58</td>
      <td>NaN</td>
      <td>22111 Hamburg Horn</td>
      <td>...</td>
      <td>['"obj_firingTypes":"district_heating"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"2"']</td>
      <td>['"obj_baseRent":"301.18"']</td>
      <td>['"obj_livingSpace":"30.89"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"y"']</td>
    </tr>
    <tr>
      <th>12494</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>101</td>
      <td>499,91</td>
      <td>in Nebenkosten enthalten</td>
      <td>['lat: 53.46145736469006,']</td>
      <td>['lng: 9.966232033537533']</td>
      <td>Denickestraße 42 b</td>
      <td>NaN</td>
      <td>21075 Hamburg Harburg</td>
      <td>...</td>
      <td>[]</td>
      <td>['"obj_hasKitchen":"n"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"2"']</td>
      <td>['"obj_baseRent":"398.91"']</td>
      <td>['"obj_livingSpace":"46.93"']</td>
      <td>['"obj_condition":"well_kept"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>


<br>
<br>

#### `df.columns`

##### Description<br>
The command `df.columns` does one thing and one thing well, one might say. It
returns a list of strings, the list of the columns of the DataFrame. Its output
can be iterated through, in order to select subsets of all columns. An iteration
can be done in the form of a list comprehension, that makes use of conditional
clauses for example. It also helps one find problematic column names, that need
to be changed in order to qualify as tidy. The output of it also helps one get
an overview of all the columns names and therefore is a starting point for
dropping certain columns and renaming the columns, so they follow an easy to
remember and precise naming pattern.

##### Example<br>
Below, the set of columns of the DataFrame are printed. One can see, that there
are two types of patterns found in the names of the columns.

1. The first set of columns originates from values found in listings, that are visible to the visitor. These ones have
   no prefix attached to them and they all have German names.
2. Columns in the second set have the prefix 'json\_' added to them. This comes from the fact, that they were sourced
   from a *script tag* found in the raw *HTML* code of each listing. The inner *HTML* of these *script tags* consisted
   of key-value pairs using a *json* like formatting. It was not machine readable though. The names of these columns
   were in English already and only the 'json\_' prefix was added afterwards.

There are several other differences between the sets, as we will see later.

```python
df.columns
```

    Index(['einbau_kue', 'lift', 'nebenkosten', 'gesamt_miete', 'heiz_kosten',
           'lat', 'lng', 'str', 'nutzf', 'regio', 'parking', 'online_since',
           'baujahr', 'objekt_zustand', 'heizungsart', 'wesent_energietr',
           'endenergiebedarf', 'kaltmiete', 'quadratmeter', 'anzahl_zimmer',
           'balkon/terasse', 'keller', 'typ', 'etage', 'anz_schlafzimmer',
           'anz_badezimmer', 'haustiere', 'nicht_mehr_verfg_seit',
           'json_heatingType', 'json_balcony', 'json_electricityBasePrice',
           'json_picturecount', 'json_telekomDownloadSpeed',
           'json_telekomUploadSpeed', 'json_totalRent', 'json_yearConstructed',
           'json_electricityKwhPrice', 'json_firingTypes', 'json_hasKitchen',
           'json_cellar', 'json_yearConstructedRange', 'json_baseRent',
           'json_livingSpace', 'json_condition', 'json_interiorQual',
           'json_petsAllowed', 'json_lift'],
          dtype='object')

<br>
<br>

#### `df.index`

##### Description<br>
The command `df.index` prints the type of the index of the DataFrame, as well as
a couple of index values from the beginning and end of the 64bit integer index
range in our example. When the final DataFrame `df` was created, using the
following line of code:

```python
df = df1.append([df2, df3], ignore_index=True)
```

The `ignore_index=True` part was important to make sure, that the range indexes
of each of the appended DataFrames `df2` and `df3` would not simply get stacked
on top of the index of `df1`. Would that have happened the resulting index would
have been unusable, since there would not have been a monotonously increasing
range index in the resulting DataFrame.

##### Example<br>
In the example it is shown what the resulting index of the final DataFrame would
have been, if parameter * ignore_index* would not have been specified at all
(`df_index_1`) and what it would have been, given `ignore_index=False`
(`df_index_2`). The resulting index is the same in both cases and it is
important, that one knows exactly how the index of any DataFrame looks like, in
order to be able to manipulate and clean it. The resulting range index of the
DataFrame, given `ignore_index=True` is used in the input statement shows all
the qualities a simple range index should have.

```python
df_index_1 = df1.append([df2, df3])
df_index_2 = df1.append([df2, df3], ignore_index=False)
print('The resulting index, if no value is specified:\n %s\n ' % df_index_1.index)
print('The resulting index, if False is used:\n %s\n ' % df_index_2.index)
```

    The resulting index, if no value is specified:
     Int64Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
                ...
                5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5591],
               dtype='int64', length=12495)

    The resulting index, if False is used:
     Int64Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
                ...
                5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5591],
               dtype='int64', length=12495)


<br>
---

```python
print('The resulting index, if True is used:\n %s\n ' % df.index)
```

    The resulting index, if True is used:
     Int64Index([    0,     1,     2,     3,     4,     5,     6,     7,     8,
                    9,
                ...
                12485, 12486, 12487, 12488, 12489, 12490, 12491, 12492, 12493,
                12494],
               dtype='int64', length=12325)

<br>
<br>

#### `df.describe()`

##### Description

The command `df.describe()` gives summary statistics for all columns, that are
of a numerical data type (*dtype*) by default. In the default case, the
following statistics are included in the output for each included column. The
following notation will be used from this point onwards: $$np.nan$$ stands for
missing values and $$\neg np.nan$$ stands for non missing values.

- **count:** Count of all $$\neg np.nan$$ for a given column.
- **mean:** The Arithmetic Mean of all $$\neg np.nan$$ in the column.
- **std:** Standard Deviation for the distribution of all $$\neg np.nan$$ in the column.
- **min:** The minimum value found in the column from the set of all $$\neg np.nan$$, also the 0% quantile.
- **25%:** Marks the 25% quantile for the distribution of $$\neg np.nan$$ in the column.
- **50%:** Like the **25%** statistic, this one sets the upper limit of the 50% quantile. Also known as the **median**.
- **75%:** The 75% quantile.
- **max:** The maximum value and the 100% quantile among all $$\neg np.nan$$ of a column.

##### Example

The *data types* (*dtypes*) in the DataFrame are checked, before `df.describe()` is explored for `df`.

```python
df.dtypes
```

    einbau_kue                    object
    lift                          object
    nebenkosten                   object
    gesamt_miete                  object
    heiz_kosten                   object
    lat                           object
    lng                           object
    str                           object
    nutzf                         object
    regio                         object
    parking                       object
    online_since                  object
    baujahr                       object
    objekt_zustand                object
    heizungsart                   object
    wesent_energietr              object
    endenergiebedarf              object
    kaltmiete                     object
    quadratmeter                  object
    anzahl_zimmer                 object
    balkon/terasse                object
    keller                        object
    typ                           object
    etage                         object
    anz_schlafzimmer             float64
    anz_badezimmer               float64
    haustiere                     object
    nicht_mehr_verfg_seit         object
    json_heatingType              object
    json_balcony                  object
    json_electricityBasePrice     object
    json_picturecount             object
    json_telekomDownloadSpeed     object
    json_telekomUploadSpeed       object
    json_totalRent                object
    json_yearConstructed          object
    json_electricityKwhPrice      object
    json_firingTypes              object
    json_hasKitchen               object
    json_cellar                   object
    json_yearConstructedRange     object
    json_baseRent                 object
    json_livingSpace              object
    json_condition                object
    json_interiorQual             object
    json_petsAllowed              object
    json_lift                     object
    dtype: object

<br>
<br>

From the output one can see, that there only 2 columns that exclusively hold
numerical data and thus have a numerical * data type* (*dtype*). All other
columns have mixed *dtypes*, so pandas labels them as having dtype 'object'. In
the following, all columns will be checked and their dtypes might change in the
process of cleaning them.

With the information, that only 2 columns have a numerical dtype, calling
`df.describe()` with no further parameters specified, will print the summary
statistics listed above only for those two columns. See the output below, for
more details.

```python
df.describe()
```

<div style="width:656px;overflow-x:scroll;">
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anz_schlafzimmer</th>
      <th>anz_badezimmer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6469.000000</td>
      <td>7317.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.579379</td>
      <td>1.096351</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.746926</td>
      <td>0.321261</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>11.000000</td>
    </tr>
  </tbody>
</table>
</div>

<br>
<br>

We will use the summary statistics for variable `anz_schlafzimmer` as an example
of how one can interpret their values for a given data series. The variable
gives the number of bedrooms that a listing has, according to the data found in
the listing on the website.

- **count:** Count of all $$\neg np.nan$$ is 6469
- **mean:** The Arithmetic Mean of all $$\neg np.nan$$ in the column is $$\bar{x} \approx 1.58$$. Since bedroom only has
  $$\neg np.nan$$ of type int64, there are no floating type numbers to be found in the column. We gained this information
  by running `df['anz_schlafzimmer'].value_counts()`, which prints a 2 column table. In the first column, all unique
  values in the data series are listed. For each of them, the count is given in the same row, second column of the
  table. $$np.nan$$ are excluded. This knowledge helps one to understand, that the mean $$\bar{x} \approx 1.58$$ signals,
  that there are many listings that have two or less bedrooms.
- **std:** Standard Deviation for the distribution of all $$\neg np.nan$$ in the data series is $$\approx 0.75$$. This gives
  the the one sigma interval, defined as $$\bar{x} \pm \sigma$$ with the standard deviation as $$\sigma$$.
- **min:** The minimum is 0, which is equivalent to no bedroom, as declared in the listing.
- **25%:** The 25% quantile reaches 1 bedroom already, so $$P(X \le 1) \le 0.25$$.
- **50%:** The value, that splits the data in two equally sized parts is 1 bedroom.
- **75%:** The 75% quantile is found at 2 bedrooms. Together with the value for the 25% quantile it is possible to
  calculate the interquartile range (**IQR**), which is given by $$Q_3 - Q_1 \equiv 2 - 1 = 1$$.
- **max:** The maximum value for the number of bedrooms found in the data series is 8.

The distributions will be analyzed at a later stage, for now the focus is on
getting a 'first look' at the DataFrame.

Below one finds the value counts for variable *anz_schlafzimmer*, which
describes the number of bedrooms found in each listing.

```python
df['anz_schlafzimmer'].value_counts()
```

    1.0    3544
    2.0    2207
    3.0     594
    4.0      97
    5.0      15
    0.0      10
    8.0       1
    6.0       1
    Name: anz_schlafzimmer, dtype: int64

#### `df.shape`

##### Description<br>
The command returns a tuple object which has two numerical values. Let $(x,y)$
be the output of `df.shape`, a tuple object where $$x$$ gives the number of rows
`df` has, while $$y$$ gives the number of columns of it.

##### Example<br>
See below for the created `df`.

```python
df.shape
```

    (12325, 47)

#### `df.count()`

##### Description<br>
`df.count()` returns the count of all $$\neg np.nan$$ for each column or for a
subset.

##### Example<br>
In the example, the output was shortened by only including 4 randomly selected
columns out of the 47 columns in the `df`.

```python
df.count().sample(4,random_state=seed)
```

    nicht_mehr_verfg_seit    11629
    json_cellar              12324
    haustiere                 3914
    json_condition           12324
    dtype: int64

<br>
<br>

#### `df.nunique()`

##### Description
Returns an integer value, that gives the number of unique values in the data
frame or of a subset of columns in the `df`. It does not return the unique
values themselves.

##### Example<br>
The example shows how it can be applied to `df` and what the output looks like.
A subset of the columns is used again, to keep the output readable.

```python
df.nunique().sample(4,random_state=seed)
```

    nicht_mehr_verfg_seit    701
    json_cellar                2
    haustiere                  3
    json_condition            10
    dtype: int64


<br>
<br>

#### `df.filter()`

##### Description<br>
`df.filter()` can be used like `df.loc`, if parameter *items* is added. It
prints the columns specified as a list of column names. However, where it shines
is when there are subsets of columns that have a certain pattern in their names.
In this case, one can use parameter *regex*, followed by a regex pattern along
with the parameter and value *axis=1*.

##### Example<br>
In the first example `df.filter()` is used like `df.loc` to select a subset of
two columns. The second example shows how *regex* can be used to filter certain
columns. As mentioned earlier, in the DataFrame constructed there is a subset of
columns, whose names all begin with the prefix 'json\_'. Using *regex*, makes it
easy to filter out these columns.

Example 1 - Using `df.filter()` to select a subset of columns.

```python
df.filter(items=['lift','str']).sample(4,random_state=seed)
```

<div style="width:656px;overflow-x:scroll;">
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lift</th>
      <th>str</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9939</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>NaN</td>
      <td>Alter Güterbahnhof 10a</td>
    </tr>
    <tr>
      <th>6023</th>
      <td>NaN</td>
      <td>Jütlandring 48</td>
    </tr>
    <tr>
      <th>12239</th>
      <td>NaN</td>
      <td>Estebogen 22</td>
    </tr>
  </tbody>
</table>
</div>



<br>
<br>

Example 2 - Using `df.filter()` to select all columns, that have the prefix 'json\_' in their names.

```python
df.filter(regex='^json', axis=1).sample(4,random_state=seed)
```

<div style="width:656px;overflow-x:scroll;">
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>json_heatingType</th>
      <th>json_balcony</th>
      <th>json_electricityBasePrice</th>
      <th>json_picturecount</th>
      <th>json_telekomDownloadSpeed</th>
      <th>json_telekomUploadSpeed</th>
      <th>json_totalRent</th>
      <th>json_yearConstructed</th>
      <th>json_electricityKwhPrice</th>
      <th>json_firingTypes</th>
      <th>json_hasKitchen</th>
      <th>json_cellar</th>
      <th>json_yearConstructedRange</th>
      <th>json_baseRent</th>
      <th>json_livingSpace</th>
      <th>json_condition</th>
      <th>json_interiorQual</th>
      <th>json_petsAllowed</th>
      <th>json_lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9939</th>
      <td>[]</td>
      <td>['"obj_balcony":"y"']</td>
      <td>['"obj_electricityBasePrice":"90.76"']</td>
      <td>['"obj_picturecount":"12"']</td>
      <td>['"obj_telekomDownloadSpeed":"16 MBit/s"']</td>
      <td>['"obj_telekomUploadSpeed":"2,4 MBit/s"']</td>
      <td>['"obj_totalRent":"1000"']</td>
      <td>['"obj_yearConstructed":"1983"']</td>
      <td>['"obj_electricityKwhPrice":"0.1985"']</td>
      <td>['"obj_firingTypes":"no_information"']</td>
      <td>['"obj_hasKitchen":"n"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"4"']</td>
      <td>['"obj_baseRent":"750"']</td>
      <td>['"obj_livingSpace":"87"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>[]</td>
      <td>['"obj_balcony":"y"']</td>
      <td>['"obj_electricityBasePrice":"90.76"']</td>
      <td>['"obj_picturecount":"5"']</td>
      <td>[]</td>
      <td>[]</td>
      <td>['"obj_totalRent":"680.23"']</td>
      <td>['"obj_yearConstructed":"2015"']</td>
      <td>['"obj_electricityKwhPrice":"0.1985"']</td>
      <td>['"obj_firingTypes":"no_information"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"n"']</td>
      <td>['"obj_yearConstructedRange":"8"']</td>
      <td>['"obj_baseRent":"441.03"']</td>
      <td>['"obj_livingSpace":"75"']</td>
      <td>['"obj_condition":"no_information"']</td>
      <td>['"obj_interiorQual":"no_information"']</td>
      <td>['"obj_petsAllowed":"no_information"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>6023</th>
      <td>['"obj_heatingType":"central_heating"']</td>
      <td>['"obj_balcony":"n"']</td>
      <td>['"obj_electricityBasePrice":"90.76"']</td>
      <td>['"obj_picturecount":"9"']</td>
      <td>[]</td>
      <td>[]</td>
      <td>['"obj_totalRent":"1188"']</td>
      <td>['"obj_yearConstructed":"1903"']</td>
      <td>['"obj_electricityKwhPrice":"0.1985"']</td>
      <td>['"obj_firingTypes":"no_information"']</td>
      <td>['"obj_hasKitchen":"y"']</td>
      <td>['"obj_cellar":"y"']</td>
      <td>['"obj_yearConstructedRange":"1"']</td>
      <td>['"obj_baseRent":"938"']</td>
      <td>['"obj_livingSpace":"67"']</td>
      <td>['"obj_condition":"well_kept"']</td>
      <td>['"obj_interiorQual":"sophisticated"']</td>
      <td>['"obj_petsAllowed":"no"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
    <tr>
      <th>12239</th>
      <td>['"obj_heatingType":"central_heating"']</td>
      <td>['"obj_balcony":"n"']</td>
      <td>['"obj_electricityBasePrice":"90.76"']</td>
      <td>['"obj_picturecount":"8"']</td>
      <td>['"obj_telekomDownloadSpeed":"100 MBit/s"']</td>
      <td>['"obj_telekomUploadSpeed":"40 MBit/s"']</td>
      <td>['"obj_totalRent":"685"']</td>
      <td>['"obj_yearConstructed":"1970"']</td>
      <td>['"obj_electricityKwhPrice":"0.1985"']</td>
      <td>['"obj_firingTypes":"oil"']</td>
      <td>['"obj_hasKitchen":"n"']</td>
      <td>['"obj_cellar":"y"']</td>
      <td>['"obj_yearConstructedRange":"2"']</td>
      <td>['"obj_baseRent":"485"']</td>
      <td>['"obj_livingSpace":"65"']</td>
      <td>['"obj_condition":"well_kept"']</td>
      <td>['"obj_interiorQual":"normal"']</td>
      <td>['"obj_petsAllowed":"negotiable"']</td>
      <td>['"obj_lift":"n"']</td>
    </tr>
  </tbody>
</table>
</div>


<br>
<br>

#### `df.sample()`

##### Description<br>
Allows one to randomly sample from a DataFrame or `pandas.Series`. It was used
several times in the examples so far, in order to give a better glimpse of the
data in the `df`. Alternatives would have been, among others, `df.head()` and
`df.tail()`. The main reason `df.sample()` was preferred over these alternatives
is the reason, that by using `df.sample()` one gets a subset of rows or columns
of the data frame, that are not constricted to either being at the very
beginning of the index in the case of `df.head()` or at the very end of the
index, if `df.tail()` is used. The subset, that `df.sample()` produces might not
have anything over the ones produced by `df.head()` or `df.tail()`, since it is
a random sample after all. It is advised to specify a value for a *seed*, that
is used used whenever any kind of random element is part of command. In the case
of `df.sample()`, one can pass a random seed in several different ways. Here,
only a integer value was needed (`seed = 42`), as defined along the imports
needed for this article. Parameter *random_state* takes the value of the random
seed (`random_state=seed`). Specifying a seed has the benefit to make the output
consistent and most importantly reproducible when run several times by one self
or by anyone else executing the file again.

This marks the end of this article. Steps from how to create a new DataFrame
from several smaller DataFrames were covered, before various tools in the pandas
library were showcased by describing each one along with using examples to show
they can be used in projects.

---
<br>
<br>
[Data Preparation Series 1]({% link _projects/data_prep_1.md %})  
[Data Preparation Series 2]({% link _projects/data_prep_2.md %})  
[Data Preparation Series 3]({% link _projects/data_prep_3.md %})  
[Data Preparation Series 4]({% link _projects/data_prep_4.md %})  
