---
layout: distill
title: 'Advanced Geospatial Feature Creation'
date: 2022-11-04
description: 'Extensive cleaning and transformation of tabular data, in order to create geospatial features. Once processed, the results are clean GPS values as "Point" objects in decimal degrees format and names of all subway and suburban train stations within Hamburg, Germany.'
img: 'assets/img/838338477938@+-67822330.jpg'
tags: ['tabular data', 'feature creation', 'geospatial feature', 'regular expression', 'data cleaning', 'data transformation', 'shapely', 'pandas', 'pyjanitor', 'geopandas', 're']
category: ['data preprocessing']
authors: 'Tobias Klein'
comments: true
---



```python
# imports
import janitor
import re
from pathlib import Path
from subprocess import check_output
import geopandas
import pandas as pd
from shapely.geometry import Point
```

# Advanced Geospatial Feature Creation

## Summary

In this article, we start by importing data of all the stations of two main types of public transport found in Hamburg, Germany. They are:
- U-Bahn (*In the following: subway*)
- S-Bahn (*In the following: suburban train*)

Using this data, the goal is to create the following features, that will be added to the list of features for every listing found in the core dataset.
- The GPS coordinates, as a tuple of latitude and longitude, using the format decimal degrees (*DD*).
- Use boolean columns to mark if the station is for subway, or for suburban trains or both.

The imported data is cleaned and the GPS coordinates are transformed for each station found in the data and features are extracted using regular expressions and the pandas library, to create new, tidy columns for the created features. All data is made compatible with the main dataset, by converting the location data of the subway and suburban train stops found in the imported flat files from degrees, minutes, and seconds (*DMS*) to DD.

Once, the new features are created, the tabular data with the new features is exported as flat files for integration with the main dataset.


## Importing The Flat Files
We create a `Path` variable `data`, that holds the path to the location where the flat files are found at.


```python
base = check_output(["pwd"]).decode("utf-8")[:-18]
base = Path(base)
print(base)
data = Path.joinpath(base, "data")
print(data)
```

    /Users/tobias/all_code/projects/python_projects/py_workflows/bachelor_thesis_portfolio_2022
    /Users/tobias/all_code/projects/python_projects/py_workflows/bachelor_thesis_portfolio_2022/data


We import two flat files. Each one holds to some degree data, that can only be found in it and that is needed to build the final public transport geospatial feature set. `c_ubahn` holds the data for all subway stops and `c_sbahn` for all suburban train stops. `low_memory=False` is added, to make sure, that pandas evaluates all rows of the DataFrame during each calculation and transformation and not just a small subset of all rows.


```python
# Reading in the data
c_ubahn = pd.read_csv(data / "ubahn_halte_mit_gps.csv", low_memory=False)
c_sbahn = pd.read_csv(data / "sbahn_halte_mit_gps.csv", low_memory=False)
```

## Initial Cleaning
We want to check the column names for both DataFrames, in order to verify that there is the same data available for both types of public transport. Upon looking at the actual headings of the DataFrames, we see that they need to be cleaned and translated first. Afterwards, the matching headings are given matching column names in both DataFrames.



```python
c_sbahn.columns, c_ubahn.columns
```




    (Index(['Bahnhof (Kürzel)Karte', 'EröffnungS-Bahn', 'Bezirk/Ort (kursiv)',
            'Kat', 'Strecke', 'Linien', 'Umstieg', 'Barriere-freier Zugang',
            'Sehenswürdigkeitenöffentliche Einrichtungen', 'Bild'],
           dtype='object'),
     Index(['Bahnhof (Kürzel)Karte', 'Linie', 'Eröffnung', 'Lage', 'Stadtteil/Ort',
            'Umsteige-möglichkeit', 'Barriere-frei', 'Anmerkungen',
            'Sehenswürdigkeitenöffent. Einrichtungen', 'Bild'],
           dtype='object'))



 We clean the headings of the two DataFrames by using the `clean_names` function from the `pyjanitor` library.


```python
c_ubahn = c_ubahn.clean_names()
c_sbahn = c_sbahn.clean_names()
c_ubahn.columns, c_sbahn.columns
```




    (Index(['bahnhof_kurzel_karte', 'linie', 'eroffnung', 'lage', 'stadtteil_ort',
            'umsteige_moglichkeit', 'barriere_frei', 'anmerkungen',
            'sehenswurdigkeitenoffent_einrichtungen', 'bild'],
           dtype='object'),
     Index(['bahnhof_kurzel_karte', 'eroffnungs_bahn', 'bezirk_ort_kursiv_', 'kat',
            'strecke', 'linien', 'umstieg', 'barriere_freier_zugang',
            'sehenswurdigkeitenoffentliche_einrichtungen', 'bild'],
           dtype='object'))



## Defining A Custom Function: Batch Processing
Since there are two DataFrames, that need the same cleaning steps, we save some time defining a function, that we hand a DataFrame as input, and returns the cleaned DataFrame. The cleaning steps it does are:
- Rename column 'bahnhof_kurzel_karte' to 'station'.
- Drop all columns except the renamed column 'station'.
- Sort the values in this column in ascending order.


```python
def drop_sort(df: pd.DataFrame):

    df = df.rename_column("bahnhof_kurzel_karte", "station")
    df.drop(
        [col for col in df.columns.tolist() if col != "station"], axis=1, inplace=True
    )
    df.sort_values("station", inplace=True)
    return df
```

Process our two DataFrames and save each returned DataFrame in a new variable.


```python
dfu = drop_sort(c_ubahn)
dfs = drop_sort(c_sbahn)
```

Check the columns of the newly created DataFrames to verify, that only column 'station' is still there.


```python
dfu.columns, dfs.columns
```




    (Index(['station'], dtype='object'), Index(['station'], dtype='object'))



Print the first five rows of each DataFrame to get a better understanding of its structure and what regular expression is needed for each, to extract the station name and the GPS coordinates from the 'station' column. One can see from the first 5 rows already, that two different regular expressions are needed for to extract the values from each one.


```python
for s in dfu, dfs:
    print(f"{s.head(5)}\n\n")
```

                                                station
    0   Ahrensburg Ost (AO)53° 39′ 41″ N, 10° 14′ 36″ O
    1  Ahrensburg West (AW)53° 39′ 52″ N, 10° 13′ 12″ O
    2        Alsterdorf (AL)53° 36′ 23″ N, 10° 0′ 42″ O
    3    Alter Teichweg (AT)53° 35′ 12″ N, 10° 3′ 52″ O
    4           Barmbek (BA)53° 35′ 14″ N, 10° 2′ 40″ O
    
    
                                                 station
    0      Agathenburg (AABG)53° 33′ 53″ N, 9° 31′ 48″ O
    1        Allermöhe (AALH)53° 29′ 25″ N, 10° 9′ 31″ O
    2  Alte Wöhr (Stadtpark) (AAW)53° 35′ 51″ N, 10° ...
    3              Altona (AAS)53° 33′ 7″ N, 9° 56′ 4″ O
    4         Aumühle (AAMS)53° 31′ 48″ N, 10° 18′ 53″ O
    
    


## The Heavy Lifting - Extracting The GPS Coordinates
While the compiled regular expression differs for the two DataFrames `dfs` and `dfu`, the steps to extract the name of the station from the 'station' column are identical. They are:
- Create a compiled regular expression pattern.
- Use pandas string method to gain access to the `str.extract` function.
    - search the column for matches.
    - extract whatever is matched by the capture group of the compiled regular expression.
    - Assign the matched portion to a new column 'name'.
- Verify the results by printing columns 'station' and 'name' to check each row in 'name' is correct.


```python
bhf = re.compile("(^.+)\s\(\w+\)")
dfs["name"] = dfs["station"].str.extract(bhf, expand=False)
```


```python
dfs.loc[:, ["station", "name"]]
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
      <th>station</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Agathenburg (AABG)53° 33′ 53″ N, 9° 31′ 48″ O</td>
      <td>Agathenburg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Allermöhe (AALH)53° 29′ 25″ N, 10° 9′ 31″ O</td>
      <td>Allermöhe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alte Wöhr (Stadtpark) (AAW)53° 35′ 51″ N, 10° ...</td>
      <td>Alte Wöhr (Stadtpark)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Altona (AAS)53° 33′ 7″ N, 9° 56′ 4″ O</td>
      <td>Altona</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aumühle (AAMS)53° 31′ 48″ N, 10° 18′ 53″ O</td>
      <td>Aumühle</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Wandsbeker Chaussee (AWCH)Lage</td>
      <td>Wandsbeker Chaussee</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Wedel (AWL)53° 34′ 55″ N, 9° 42′ 18″ O</td>
      <td>Wedel</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Wellingsbüttel (AWBS)53° 38′ 28″ N, 10° 4′ 57″ O</td>
      <td>Wellingsbüttel</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Wilhelmsburg (AWFS)53° 29′ 53″ N, 10° 0′ 24″ O</td>
      <td>Wilhelmsburg</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Wohltorf (AWLF)53° 31′ 14″ N, 10° 16′ 40″ O</td>
      <td>Wohltorf</td>
    </tr>
  </tbody>
</table>
<p>68 rows × 2 columns</p>
</div>




```python
bhf_ubahn = re.compile("(^[-a-zA-Züäöß.ÜÖ,Ä]{0,30}\s?[-a-zA-Züäöß.ÜÖ,Ä]{0,30}[^(\d])")
dfu["name"] = dfu["station"].str.extract(bhf_ubahn, expand=False)
```


```python
dfu.loc[:, ["station", "name"]]
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
      <th>station</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ahrensburg Ost (AO)53° 39′ 41″ N, 10° 14′ 36″ O</td>
      <td>Ahrensburg Ost</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ahrensburg West (AW)53° 39′ 52″ N, 10° 13′ 12″ O</td>
      <td>Ahrensburg West</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alsterdorf (AL)53° 36′ 23″ N, 10° 0′ 42″ O</td>
      <td>Alsterdorf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alter Teichweg (AT)53° 35′ 12″ N, 10° 3′ 52″ O</td>
      <td>Alter Teichweg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barmbek (BA)53° 35′ 14″ N, 10° 2′ 40″ O</td>
      <td>Barmbek</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Wandsbek Markt (WM)53° 34′ 19″ N, 10° 4′ 4″ O</td>
      <td>Wandsbek Markt</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Wandsbek-Gartenstadt (WK)53° 35′ 32″ N, 10° 4′...</td>
      <td>Wandsbek-Gartenstadt</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Wandsbeker Chaussee (WR)53° 34′ 12″ N, 10° 3′ ...</td>
      <td>Wandsbeker Chaussee</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Wartenau (WA)53° 33′ 51″ N, 10° 2′ 4″ O</td>
      <td>Wartenau</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Überseequartier (UR)53° 32′ 26″ N, 9° 59′ 56″ O</td>
      <td>Überseequartier</td>
    </tr>
  </tbody>
</table>
<p>99 rows × 2 columns</p>
</div>



## Extract Entire Coordinate Pairs
Given, that the coordinates have format *DMS* in the input files, regular expressions are used to extract the entire coordinate pair for each station. After looking at the value range for the *minute* component across all rows, there can only be one value for the minute component, that is *53*. The pattern matches and captures everything until the last capital *O*, which there is only one, which marks the end of one complete coordinate pair.


```python
# five = re.compile("([4-6][1-6].+O)")
five = re.compile("(53.+O)")
dfu["gps"] = dfu["station"].str.extract(five, expand=False)
dfs["gps"] = dfs["station"].str.extract(five, expand=False)
```

## Check For Missing Values
For the subway stations, there are no missing values in the GPS column. Which means, that the capture group extracted something for all rows.


```python
dfu[dfu.gps.isna() == True]
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
      <th>station</th>
      <th>name</th>
      <th>gps</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



While there are two missing values found in the table for the suburban trains, it shows that these missing values were not introduced as a result of the regular expressions pattern not matching the coordinates for two rows in the DataFrame. We can see that there are no coordinate values found in the 'station' column and that is why rows 25 and 63 in the GPS column have missing values. This will be fixed in the next step.


```python
dfs[dfs.gps.isna() == True]
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
      <th>station</th>
      <th>name</th>
      <th>gps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>Hauptbahnhof (AHS)Lage</td>
      <td>Hauptbahnhof</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Wandsbeker Chaussee (AWCH)Lage</td>
      <td>Wandsbeker Chaussee</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Fill Missing Values
The missing values are for stations at 'Hauptbahnhof' and 'Wandsbeker Chaussee'. We look for other stations at these locations, that have GPS values. Their GPS coordinates should be close to those of the missing stations.

No other entries, apart from the two rows with missing GPS coordinates are found in the suburban train stations DataFrame. Next, we try to find rows in the subway stations DataFrame, that have valid GPS coordinates for the two stations.


```python
dfs[dfs["station"].str.contains(r"Haupt|Wandsbek")]
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
      <th>station</th>
      <th>name</th>
      <th>gps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>Hauptbahnhof (AHS)Lage</td>
      <td>Hauptbahnhof</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Wandsbeker Chaussee (AWCH)Lage</td>
      <td>Wandsbeker Chaussee</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Valid values for both stations are found in the subways stations DataFrame. Row indexes of these rows are 34 for 'Hauptbahnhof' and 97 for 'Wandsbeker Chaussee'. These indexes are used to fill the missing values.


```python
dfu[dfu["station"].str.contains("Haupt|Wandsbek")]
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
      <th>station</th>
      <th>name</th>
      <th>gps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>Hauptbahnhof Nord (HX)53° 33′ 15″ N, 10° 0′ 25″ O</td>
      <td>Hauptbahnhof Nord</td>
      <td>53° 33′ 15″ N, 10° 0′ 25″ O</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Hauptbahnhof Süd (HB)53° 33′ 8″ N, 10° 0′ 30″ O</td>
      <td>Hauptbahnhof Süd</td>
      <td>53° 33′ 8″ N, 10° 0′ 30″ O</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Wandsbek Markt (WM)53° 34′ 19″ N, 10° 4′ 4″ O</td>
      <td>Wandsbek Markt</td>
      <td>53° 34′ 19″ N, 10° 4′ 4″ O</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Wandsbek-Gartenstadt (WK)53° 35′ 32″ N, 10° 4′...</td>
      <td>Wandsbek-Gartenstadt</td>
      <td>53° 35′ 32″ N, 10° 4′ 27″ O</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Wandsbeker Chaussee (WR)53° 34′ 12″ N, 10° 3′ ...</td>
      <td>Wandsbeker Chaussee</td>
      <td>53° 34′ 12″ N, 10° 3′ 31″ O</td>
    </tr>
  </tbody>
</table>
</div>



Using the row indexes in the subway stations DataFrame from above, the missing values in the suburban train stations DataFrame are filled manually by overwriting their values in the `gps` column.


```python
dfs.loc[25, "gps"] = dfu.loc[34, "gps"]
dfs.loc[63, "gps"] = dfu.loc[97, "gps"]
```

Splitting on ',', we can create two new columns from the values in the `gps` column. The new columns will hold the latitude values and the longitude values respectively for each station.


```python
dfu[["lat", "lng"]] = dfu["gps"].str.split(",", expand=True)
dfs[["lat", "lng"]] = dfs["gps"].str.split(",", expand=True)
```

Before defining the function to convert the GPS coordinates from DMS to DD format, we look at one row in the latitude and longitude columns of both DataFrames. The structure of the values looks identically in both, that means that there should be no need to define two separate conversion functions, one should suffice.


```python
def prev(df: pd.DataFrame):
    print(
        f"latitude component looks like: {df.lat[49]},\n\nlongitude component looks like {df.lng[49]}"
    )
```


```python
prev(dfu)
```

    latitude component looks like: 53° 39′ 39″ N,
    
    longitude component looks like  10° 1′ 3″ O



```python
prev(dfs)
```

    latitude component looks like: 53° 39′ 7″ N,
    
    longitude component looks like  10° 5′ 38″ O


In order to convert the GPS values from DMS to DD, we define functions, that will do the conversion for us. We test the output of the function to make sure the output is as expected.


```python
def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
    if direction == "E":
        dd *= -1
    return dd

def parse_dms(dms):
    parts = re.split("[^\d\w\.]+", dms)
    lat = dms2dd(parts[0], parts[1], parts[2], parts[3])
    return lat

dd = parse_dms("78°55'44.33324\" N")
print(dd)
```

    78.92898145555556



```python
print(dfs.lat[4])
print(f"{parse_dms(dfs.lat[4])}")
```

    53° 31′ 48″ N
    53.53


All rows in the `lat` and `lng` columns are stripped of any remaining white space, if there is any to be found. This is done prior to the `parse_dms` function being applied, since the regex pattern might fail to match all relevant parts of the coordinates, if there was any white space to be found, that was not accounted for.


```python
dfu["lng"] = dfu["lng"].str.strip()
dfu["lat"] = dfu["lat"].str.strip()
dfs["lng"] = dfs["lng"].str.strip()
dfs["lat"] = dfs["lat"].str.strip()
```

The custom conversion function `parse_dms` is applied to all rows in the `lat` and `lng` columns of both DataFrames respectively and the results are saved in new columns, that share the same suffix `_dd`, alias for *decimal degrees* format. The reason for creating these new columns, is that we want to be able to compare the values before and after the conversion in each row of the DataFrames.


```python
dfu["lat_dd"] = dfu["lat"].apply(lambda x: parse_dms(x))
dfu["lng_dd"] = dfu["lng"].apply(lambda x: parse_dms(x))
dfs["lat_dd"] = dfs["lat"].apply(lambda x: parse_dms(x))
dfs["lng_dd"] = dfs["lng"].apply(lambda x: parse_dms(x))
```

## Creating The Final GPS Column
Using the `assign` function from the pandas library, a new column `white space` is created. Its values are tuples of GPS coordinates for each subway and suburban train station found in the data. This step is needed in order to apply the `Point` conversion from the `shapely.geometry` module. This conversion makes it possible to integrate the features created here into the listings data in the core dataset. It enables us to compute distances between listings and any station found in the two datasets processed here. The ability to compute distances between features gives a powerful tool, that can enhance the results we get from interpreting a tree based model in the later stages of this project for example.


```python
dfs = dfs.assign(
    gps_dd=list(dfs[["lng_dd", "lat_dd"]].itertuples(index=False, name=None))
)
dfu = dfu.assign(
    gps_dd=list(dfu[["lng_dd", "lat_dd"]].itertuples(index=False, name=None))
)
```

Applying the `Point` conversion can throw a lot of errors, that would require us to look at all the previous steps and carefully apply the transformations again possibly. The documentation of the `shapely` library has been the best source of knowledge for me to find answers to problems encountered when trying to apply the `Point` conversion.


```python
dfs["gps_dd"] = dfs["gps_dd"].apply(Point)
dfu["gps_dd"] = dfu["gps_dd"].apply(Point)
```

```json
[
    {
        "station": [
            "\"Ahrensburg Ost (AO)53° 39′ 41″ N, 10° 14′ 36″ O\"",
            "\"Ahrensburg West (AW)53° 39′ 52″ N, 10° 13′ 12″ O\"",
            "\"Alsterdorf (AL)53° 36′ 23″ N, 10° 0′ 42″ O\"",
            "\"Alter Teichweg (AT)53° 35′ 12″ N, 10° 3′ 52″ O\"",
            "\"Barmbek (BA)53° 35′ 14″ N, 10° 2′ 40″ O\"",
            "\"Baumwall (BL)53° 32′ 39″ N, 9° 58′ 54″ O\""
        ]
    },
    {
        "name": [
            "Ahrensburg Ost",
            "Ahrensburg West",
            "Alsterdorf",
            "Alter Teichweg",
            "Barmbek",
            "Baumwall"
        ]
    },
    {
        "gps": [
            "\"53° 39′ 41″ N, 10° 14′ 36″ O\"",
            "\"53° 39′ 52″ N, 10° 13′ 12″ O\"",
            "\"53° 36′ 23″ N, 10° 0′ 42″ O\"",
            "\"53° 35′ 12″ N, 10° 3′ 52″ O\"",
            "\"53° 35′ 14″ N, 10° 2′ 40″ O\"",
            "\"53° 32′ 39″ N, 9° 58′ 54″ O\""
        ]
    },
    {
        "lat": [
            "53° 39′ 41″ N",
            "53° 39′ 52″ N",
            "53° 36′ 23″ N",
            "53° 35′ 12″ N",
            "53° 35′ 14″ N",
            "53° 32′ 39″ N"
        ]
    },
    {
        "lng": [
            "10° 14′ 36″ O",
            "10° 13′ 12″ O",
            "10° 0′ 42″ O",
            "10° 3′ 52″ O",
            "10° 2′ 40″ O",
            "9° 58′ 54″ O"
        ]
    },
    {
        "lat_dd": [
            "53.66138888888889",
            "53.66444444444444",
            "53.60638888888889",
            "53.586666666666666",
            "53.58722222222222",
            "53.54416666666666"
        ]
    },
    {
        "lng_dd": [
            "10.243333333333332",
            "10.22",
            "10.011666666666667",
            "10.064444444444446",
            "10.044444444444444",
            "9.981666666666667"
        ]
    },
    {
        "gps_dd": [
            "POINT (10.243333333333332 53.66138888888889)",
            "POINT (10.22 53.66444444444444)",
            "POINT (10.011666666666667 53.60638888888889)",
            "POINT (10.064444444444446 53.586666666666666)",
            "POINT (10.044444444444444 53.58722222222222)",
            "POINT (9.981666666666667 53.54416666666666)"
        ]
    }
]

```

## Exporting The Cleaned Station Data
The two tables are ready to be integrated into to main dataset, and are therefore exported as csv files.


```python
dfs.to_csv("../data/s-bahn_final.csv")
dfu.to_csv("../data/u-bahn_final.csv")
```

These are the steps for creating new geospatial features, that are completely independent of the ones in the core dataset scraped from www.immobilienscout24.de . In the following steps, these features will be integrated into the core dataset and used to create new features for each listing found in the core dataset.
