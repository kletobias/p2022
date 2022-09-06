---
layout: distill
title: 'Cleaning a 47 Column<br>Pandas DataFrame<br>Part 4'
description: 'Showcase of how batch processing several columns of tabular data using pandas, pyjanitor and the re library can look like.'
img: 'assets/img/838338477938@+-3948324823.jpg'
importance: 4
tags: tabular data, pandas, batch processing, data validation using regular expressions
category: data preprocessing
---

### Summary Of This Article
Showcase of how batch processing several columns in a tabular dataset, using
`pandas`, `pyjanitor` and the `re` library can look like. Regular expressions
are used to transform columns with messy data into ones with valid row contents.
Redundant columns are dropped, columns are reordered by type. Columns with dtype
`categorical` are created and their classes converted to numerical values for
the following evaluation of candidate models.<br>
<br>

# Wrangling with that Data! 4/4

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

## Preparations for the Batch Operations
The unique values are calculated for all columns, that are still to be cleaned and give the basis for the regex patterns used in the following steps to clean these remaining columns. The output with all the unique values is truncated for readability.


```python
col_stats = {}

for n in df.columns:
    col_stats[n] = {}
    try:
        col_stats[n]["unique"] = df[n].unique()
    except TypeError:
        pass

for col in df.columns:
    try:
        ut = col_stats[col]["unique"][:5] if len(col_stats[col]["unique"]) > 80 else col_stats[col]["unique"]
        print(f'{col}:\n{ut}\n\n')
    except KeyError:
        pass
```

    bfitted_kitchen:
    [1 0]


    belevator:
    [0 1]


    auxiliary_costs:
    [190.84 117.95 249.83  70.   213.33]


    total_rent:
    [541.06 559.05 839.01 665.   651.81]


    lat:
    [53.52943934 53.58414267 53.60449188 53.5639497  53.60180649]


    lng:
    [10.15235715 10.06842725  9.86423116  9.95688142 10.08125725]


    parking:
    [nan ' Tiefgaragen-Stellplatz ' ' 3 Tiefgaragen-Stellplätze '
     ' 1 Duplex-Stellplatz ' ' 1 Tiefgaragen-Stellplatz '
     ' 1 Außenstellplatz ' ' 1 Garage ' ' 2 Tiefgaragen-Stellplätze '
     ' 25 Tiefgaragen-Stellplätze ' ' 1 Stellplatz ' ' Carport '
     ' 2 Stellplätze ' ' 1 Carport ' ' Außenstellplatz '
     ' Parkhaus-Stellplatz ' ' Duplex-Stellplatz ' ' Garage '
     ' 2 Parkhaus-Stellplätze ' ' 2 Garagen ' ' 9 Stellplätze '
     ' 2 Außenstellplätze ' ' 6 Außenstellplätze ' ' 2 Carports '
     ' 1 Parkhaus-Stellplatz ' ' 20 Stellplätze '
     ' 12 Tiefgaragen-Stellplätze ' ' 6 Garagen '
     ' 6 Tiefgaragen-Stellplätze ' ' 5 Außenstellplätze ']


    date_listed:
    ['2018-12-03T00:00:00.000000000' '2018-12-02T00:00:00.000000000'
     '2018-11-30T00:00:00.000000000' '2018-11-29T00:00:00.000000000'
     '2018-11-28T00:00:00.000000000']


    yoc:
    ['1976' '1950' '1997' '1951' '2001']


    object_condition:
    [nan ' Gepflegt ' ' Vollständig renoviert ' ' Erstbezug '
     ' Erstbezug nach Sanierung ' ' Neuwertig ' ' Modernisiert '
     ' Renovierungsbedürftig ' ' Saniert ' ' Nach Vereinbarung ']


    heating_type:
    [nan ' Etagenheizung ' ' Zentralheizung ' ' Fernwärme '
     ' Elektro-Heizung ' ' Nachtspeicheröfen ' ' Gas-Heizung '
     ' Blockheizkraftwerke ' ' Fußbodenheizung ' ' Öl-Heizung '
     ' Holz-Pelletheizung ' ' Ofenheizung ' ' Wärmepumpe ']


    main_es:
    [' Fernwärme ' nan ' Öl ' ' Strom ' ' Gas, Strom ' ' Gas ' ' KWK fossil '
     ' Erdgas schwer ' ' Erdgas leicht ' ' Gas, Fernwärme ' ' Nahwärme '
     ' Fernwärme, Strom ' ' Erdwärme ' ' Fernwärme-Dampf ' ' Öl, Strom '
     ' Flüssiggas ' ' Wärmelieferung ' ' KWK regenerativ ' ' Holzpellets '
     ' Erdwärme, Strom ' ' Solar, Gas ' ' Öl, Fernwärme '
     ' Fernwärme, Erdgas schwer ' ' Erdwärme, Fernwärme ' ' Solar '
     ' Gas, Öl ' ' Umweltwärme ' ' Strom, Erdgas leicht ' ' KWK erneuerbar ']


    total_energy_need:
    [nan ' 132,9 kWh/(m²*a) ' ' 65,9 kWh/(m²*a) ' ' 49 kWh/(m²*a) '
     ' 50 kWh/(m²*a) ']


    base_rent:
    [' 350,22 € ' ' 441,10 € ' ' 589,18 € ' ' 595 € ' ' 438,48 € ']


    square_meters:
    [' 76 m² ' ' 60 m² ' ' 75 m² ' ' 46 m² ' ' 52 m² ']


    no_rooms:
    [' 3 ' ' 2,5 ' ' 2 ' ' 1,5 ' ' 3,5 ' ' 1 ' ' 4 ' ' 4,5 ' ' 5 ' ' 6 '
     ' 5,5 ' ' 6,5 ' ' 7,5 ' ' 8 ' ' 7 ' ' 36 ' ' 140 ']


    bbalcony:
    [nan 'Balkon/ Terrasse']


    cellar:
    [nan 'Keller']


    type:
    [nan ' Sonstige ' ' Dachgeschoss ' ' Etagenwohnung ' ' Hochparterre '
     ' Erdgeschosswohnung ' ' Maisonette ' ' Souterrain ' ' Terrassenwohnung '
     ' Penthouse ' ' Loft ']


    floor:
    [' 0 ' ' 3 ' ' 2 ' nan ' 1 ']


    no_bedrooms:
    [nan  1.  2.  3.  4.  0.  5.  6.]


    no_bathrooms:
    [nan  1.  2.  0.  3. 11.]


    bpets_allowed:
    [nan ' Ja ' ' Nach Vereinbarung ' ' Nein ']


    date_unlisted:
    ['2018-12-03T00:00:00.000000000' '2018-12-02T00:00:00.000000000'
     '2018-11-30T00:00:00.000000000' '2018-12-01T00:00:00.000000000'
     '2018-11-29T00:00:00.000000000']


    json_heating_type:
    [nan '[\'"obj_heatingType":"self_contained_central_heating"\']'
     '[\'"obj_heatingType":"central_heating"\']'
     '[\'"obj_heatingType":"no_information"\']'
     '[\'"obj_heatingType":"district_heating"\']'
     '[\'"obj_heatingType":"electric_heating"\']'
     '[\'"obj_heatingType":"night_storage_heater"\']'
     '[\'"obj_heatingType":"gas_heating"\']'
     '[\'"obj_heatingType":"combined_heat_and_power_plant"\']'
     '[\'"obj_heatingType":"floor_heating"\']'
     '[\'"obj_heatingType":"oil_heating"\']'
     '[\'"obj_heatingType":"wood_pellet_heating"\']'
     '[\'"obj_heatingType":"stove_heating"\']'
     '[\'"obj_heatingType":"heat_pump"\']']


    json_balcony:
    ['[\'"obj_balcony":"n"\']' '[\'"obj_balcony":"y"\']']


    json_picturecount:
    ['[\'"obj_picturecount":"6"\']' '[\'"obj_picturecount":"11"\']'
     '[\'"obj_picturecount":"5"\']' '[\'"obj_picturecount":"0"\']'
     '[\'"obj_picturecount":"13"\']' '[\'"obj_picturecount":"2"\']'
     '[\'"obj_picturecount":"4"\']' '[\'"obj_picturecount":"7"\']'
     '[\'"obj_picturecount":"3"\']' '[\'"obj_picturecount":"10"\']'
     '[\'"obj_picturecount":"1"\']' '[\'"obj_picturecount":"9"\']'
     '[\'"obj_picturecount":"8"\']' '[\'"obj_picturecount":"15"\']'
     '[\'"obj_picturecount":"12"\']' '[\'"obj_picturecount":"24"\']'
     '[\'"obj_picturecount":"17"\']' '[\'"obj_picturecount":"14"\']'
     '[\'"obj_picturecount":"21"\']' '[\'"obj_picturecount":"18"\']'
     '[\'"obj_picturecount":"19"\']' '[\'"obj_picturecount":"25"\']'
     '[\'"obj_picturecount":"20"\']' '[\'"obj_picturecount":"22"\']'
     '[\'"obj_picturecount":"40"\']' '[\'"obj_picturecount":"29"\']'
     '[\'"obj_picturecount":"16"\']' '[\'"obj_picturecount":"28"\']'
     '[\'"obj_picturecount":"27"\']' '[\'"obj_picturecount":"36"\']'
     '[\'"obj_picturecount":"35"\']' '[\'"obj_picturecount":"34"\']'
     '[\'"obj_picturecount":"26"\']' '[\'"obj_picturecount":"23"\']'
     '[\'"obj_picturecount":"43"\']' '[\'"obj_picturecount":"32"\']'
     '[\'"obj_picturecount":"31"\']' '[\'"obj_picturecount":"38"\']'
     '[\'"obj_picturecount":"30"\']' '[\'"obj_picturecount":"49"\']'
     '[\'"obj_picturecount":"39"\']' '[\'"obj_picturecount":"37"\']'
     '[\'"obj_picturecount":"42"\']' '[\'"obj_picturecount":"41"\']'
     '[\'"obj_picturecount":"45"\']' '[\'"obj_picturecount":"64"\']']


    json_telekomdownloadspeed:
    ['[\'"obj_telekomDownloadSpeed":"100 MBit/s"\']'
     '[\'"obj_telekomDownloadSpeed":"50 MBit/s"\']'
     '[\'"obj_telekomDownloadSpeed":"16 MBit/s"\']' nan
     '[\'"obj_telekomDownloadSpeed":"25 MBit/s"\']'
     '[\'"obj_telekomDownloadSpeed":"6 MBit/s"\']'
     '[\'"obj_telekomDownloadSpeed":"200 MBit/s"\']']


    json_total_rent:
    ['[\'"obj_totalRent":"541.06"\']' '[\'"obj_totalRent":"559.05"\']'
     '[\'"obj_totalRent":"839.01"\']' '[\'"obj_totalRent":"665"\']'
     '[\'"obj_totalRent":"651.81"\']']


    json_yoc:
    ['[\'"obj_yearConstructed":"1976"\']' '[\'"obj_yearConstructed":"1950"\']'
     '[\'"obj_yearConstructed":"1997"\']' '[\'"obj_yearConstructed":"1951"\']'
     '[\'"obj_yearConstructed":"2001"\']']


    json_main_es:
    ['[\'"obj_firingTypes":"district_heating"\']'
     '[\'"obj_firingTypes":"no_information"\']'
     '[\'"obj_firingTypes":"oil"\']' '[\'"obj_firingTypes":"electricity"\']'
     nan '[\'"obj_firingTypes":"gas"\']'
     '[\'"obj_firingTypes":"natural_gas_heavy"\']'
     '[\'"obj_firingTypes":"natural_gas_light"\']'
     '[\'"obj_firingTypes":"local_heating"\']'
     '[\'"obj_firingTypes":"geothermal"\']'
     '[\'"obj_firingTypes":"liquid_gas"\']'
     '[\'"obj_firingTypes":"heat_supply"\']'
     '[\'"obj_firingTypes":"pellet_heating"\']'
     '[\'"obj_firingTypes":"solar_heating"\']']


    json_bfitted_kitchen:
    ['[\'"obj_hasKitchen":"y"\']' '[\'"obj_hasKitchen":"n"\']']


    json_cellar:
    ['[\'"obj_cellar":"n"\']' '[\'"obj_cellar":"y"\']']


    json_const_time:
    ['[\'"obj_yearConstructedRange":"3"\']'
     '[\'"obj_yearConstructedRange":"1"\']'
     '[\'"obj_yearConstructedRange":"5"\']'
     '[\'"obj_yearConstructedRange":"2"\']'
     '[\'"obj_yearConstructedRange":"6"\']'
     '[\'"obj_yearConstructedRange":"4"\']' nan
     '[\'"obj_yearConstructedRange":"8"\']'
     '[\'"obj_yearConstructedRange":"7"\']'
     '[\'"obj_yearConstructedRange":"9"\']']


    json_base_rent:
    ['[\'"obj_baseRent":"350.22"\']' '[\'"obj_baseRent":"441.1"\']'
     '[\'"obj_baseRent":"589.18"\']' '[\'"obj_baseRent":"595"\']'
     '[\'"obj_baseRent":"438.48"\']']


    json_square_meters:
    ['[\'"obj_livingSpace":"76"\']' '[\'"obj_livingSpace":"60"\']'
     '[\'"obj_livingSpace":"75"\']' '[\'"obj_livingSpace":"46"\']'
     '[\'"obj_livingSpace":"52"\']']


    json_object_condition:
    ['[\'"obj_condition":"no_information"\']'
     '[\'"obj_condition":"well_kept"\']'
     '[\'"obj_condition":"fully_renovated"\']'
     '[\'"obj_condition":"first_time_use"\']'
     '[\'"obj_condition":"first_time_use_after_refurbishment"\']'
     '[\'"obj_condition":"mint_condition"\']'
     '[\'"obj_condition":"modernized"\']'
     '[\'"obj_condition":"need_of_renovation"\']'
     '[\'"obj_condition":"refurbished"\']'
     '[\'"obj_condition":"negotiable"\']']


    json_interiorqual:
    ['[\'"obj_interiorQual":"no_information"\']'
     '[\'"obj_interiorQual":"normal"\']'
     '[\'"obj_interiorQual":"sophisticated"\']'
     '[\'"obj_interiorQual":"simple"\']' '[\'"obj_interiorQual":"luxury"\']']


    json_bpets_allowed:
    ['[\'"obj_petsAllowed":"no_information"\']'
     '[\'"obj_petsAllowed":"yes"\']' '[\'"obj_petsAllowed":"negotiable"\']'
     '[\'"obj_petsAllowed":"no"\']']


    json_belevator:
    ['[\'"obj_lift":"n"\']' '[\'"obj_lift":"y"\']']


    time_listed:
    [              0 259200000000000  86400000000000 172800000000000
     518400000000000]




## Applying The Batch Processing

The steps below are all chosen by the unique values of each included column, so
that the result of this batch processing procedure takes care of the problems
found in these columns.


```python
df = (
        df.process_text(
                column_name="base_rent",
                string_function="extract",
                pat=r"([\d,.]+)",
                expand=False,
                )
            .process_text(
                column_name="square_meters",
                string_function="extract",
                pat=r"([\d.,]+)",
                expand=False,
                )
            .process_text(
                column_name="square_meters", string_function="replace", pat=",", repl="."
                )
            .process_text(
                column_name="floor",
                string_function="extract",
                pat=r"[^\d]+?(-?\d{1,2})",
                expand=False,
                )
            .process_text(
                column_name="no_rooms",
                string_function="extract",
                pat=r"[^\d]+?(-?\d{1},?\d?)",
                expand=False,
                )
            .process_text(column_name="no_rooms", string_function="replace", pat=",", repl=".")
            .process_text(
                column_name="total_energy_need",
                string_function="extract",
                pat=r"(\d+,?\d+?)",
                expand=False,
                )
            .process_text(
                column_name="total_energy_need", string_function="replace", pat=",", repl="."
                )
            .process_text(
                column_name="parking", string_function="replace", pat=r"\d+?", repl=""
                )
            .process_text(
                column_name="base_rent",
                string_function="replace",
                pat=r"(\d{1,2})\.(\d{3})",
                repl=r"\1\2",
                )
            .process_text(column_name="base_rent", string_function="replace", pat=",", repl=".")
            .find_replace(
                match="exact", yoc={
                        "unbekannt": "no_information"
                        }, no_rooms={
                        ",": "."
                        }
                )
            .process_text(
                column_name="json_heating_type",
                string_function="extract",
                pat=r':"([a-z_]+)"',
                expand=False,
                )
            .process_text(
                column_name="json_balcony",
                string_function="extract",
                pat=r'"([a-z])"',
                expand=False,
                )
            .process_text(
                column_name="json_picturecount",
                string_function="extract",
                pat=r"(\d{1,2})",
                expand=False,
                )
            .process_text(
                column_name="json_total_rent",
                string_function="extract",
                pat=r"([0-9.]+)",
                expand=False,
                )
            .process_text(
                column_name="json_yoc", string_function="extract", pat=r"(\d{4})", expand=False
                )
            .process_text(
                column_name="json_main_es",
                string_function="extract",
                pat=r':"([a-z_]+)"',
                expand=False,
                )
            .process_text(
                column_name="json_bfitted_kitchen",
                string_function="extract",
                pat=r'"([a-z])"',
                expand=False,
                )
            .process_text(
                column_name="json_cellar",
                string_function="extract",
                pat=r'"([a-z])"',
                expand=False,
                )
            .process_text(
                column_name="json_const_time",
                string_function="extract",
                pat=r'"(\d)"',
                expand=False,
                )
            .process_text(
                column_name="json_telekomdownloadspeed",
                string_function="extract",
                pat=r'(\d{1,3})',
                expand=False,
                )
            .process_text(
                column_name="json_base_rent",
                string_function="extract",
                pat=r'"([\d.]+)"',
                expand=False,
                )
            .process_text(
                column_name="json_square_meters",
                string_function="extract",
                pat=r'"([\d.]+)"',
                expand=False,
                )
            .process_text(
                column_name="json_object_condition",
                string_function="extract",
                pat=r':"([a-z_]+)"',
                expand=False,
                )
            .process_text(
                column_name="json_interiorqual",
                string_function="extract",
                pat=r':"([a-z_]+)"',
                expand=False,
                )
            .process_text(
                column_name="json_bpets_allowed",
                string_function="extract",
                pat=r':"([a-z_]+)"',
                expand=False,
                )
            .process_text(
                column_name="json_belevator",
                string_function="extract",
                pat=r'"([a-z])"',
                expand=False,
                )
            .fill_empty(
                column_names=[
                        col
                        for col in df.columns
                        if df[col].dtype not in ["timedelta64[ns]"] and col != "gps"
                        ],
                value=-9999,
                )
            .rename_column("json_balcony", "json_bbalcony")
)
```



The columns with counterparts are specified and the columns without the `json_`
prefix are overwritten by the identical values of their `json_` counterparts,
after the unique values for each column pair are checked for differing values.
The columns with a  `json_` prefix have values, that need less cleaning compared
to the ones found in the non-json prefix columns, hence the values found in the
`json_` columns are kept.


```python
json_check = [
        "heating_type",
        "bbalcony",
        "cellar",
        "heating_type",
        "total_rent",
        "yoc",
        "main_es",
        "bfitted_kitchen",
        "base_rent",
        "square_meters",
        "object_condition",
        "bpets_allowed",
        "belevator",
        ]
json_colnames = []
for col in json_check:
    json_colnames.append(f"json_{col}")
    assert col in df.columns
    assert f"json_{col}" in df.columns

for cc in zip(json_check, json_colnames):
    col_stats["json_check"] = {}
    col_stats["json_check"][cc[0]] = df.loc[:, [cc[0], cc[1]]].value_counts()
    ut = col_stats["json_check"][cc[0]][0:1] if len(col_stats["json_check"][cc[0]]) > 20 else col_stats["json_check"][cc[0]]
    print("\n", ut, "\n")
```


     heating_type           json_heating_type             
     Zentralheizung        central_heating                   4763
    -9999                  -9999                             2074
     Fernwärme             district_heating                   966
     Etagenheizung         self_contained_central_heating     449
     Gas-Heizung           gas_heating                        351
    -9999                  no_information                     293
     Fußbodenheizung       floor_heating                      230
     Blockheizkraftwerke   combined_heat_and_power_plant      105
     Öl-Heizung            oil_heating                         70
     Nachtspeicheröfen     night_storage_heater                63
     Holz-Pelletheizung    wood_pellet_heating                 30
     Elektro-Heizung       electric_heating                    16
     Wärmepumpe            heat_pump                            9
     Ofenheizung           stove_heating                        4
    dtype: int64


     bbalcony          json_bbalcony
    Balkon/ Terrasse  y                6400
    -9999             n                3023
    dtype: int64


     cellar  json_cellar
    Keller  y              4873
    -9999   n              4550
    dtype: int64


     heating_type           json_heating_type             
     Zentralheizung        central_heating                   4763
    -9999                  -9999                             2074
     Fernwärme             district_heating                   966
     Etagenheizung         self_contained_central_heating     449
     Gas-Heizung           gas_heating                        351
    -9999                  no_information                     293
     Fußbodenheizung       floor_heating                      230
     Blockheizkraftwerke   combined_heat_and_power_plant      105
     Öl-Heizung            oil_heating                         70
     Nachtspeicheröfen     night_storage_heater                63
     Holz-Pelletheizung    wood_pellet_heating                 30
     Elektro-Heizung       electric_heating                    16
     Wärmepumpe            heat_pump                            9
     Ofenheizung           stove_heating                        4
    dtype: int64


     total_rent  json_total_rent
    750.0       750                49
    dtype: int64


     yoc    json_yoc
    -9999  -9999       599
    dtype: int64


     main_es  json_main_es
     Gas     gas             3257
    dtype: int64


     bfitted_kitchen  json_bfitted_kitchen
    1                y                       5755
    0                n                       3668
    dtype: int64


     base_rent  json_base_rent
    650        650               112
    dtype: int64


     square_meters  json_square_meters
    60             60                    157
    dtype: int64


     object_condition            json_object_condition             
    -9999                       no_information                        3617
     Gepflegt                   well_kept                             3047
     Neuwertig                  mint_condition                         639
     Erstbezug                  first_time_use                         554
     Modernisiert               modernized                             465
     Erstbezug nach Sanierung   first_time_use_after_refurbishment     400
     Vollständig renoviert      fully_renovated                        375
     Saniert                    refurbished                            279
     Renovierungsbedürftig      need_of_renovation                      36
     Nach Vereinbarung          negotiable                              11
    dtype: int64


     bpets_allowed        json_bpets_allowed
    -9999                no_information        6467
     Nach Vereinbarung   negotiable            1673
     Nein                no                    1124
     Ja                  yes                    159
    dtype: int64


     belevator  json_belevator
    0          n                 7420
    1          y                 2003
    dtype: int64




```python
df.drop(columns=json_check, inplace=True)
```

## Removal Of Any Remaining White Space


```python
for col in df.columns:
    if col not in ["gps", "time_listed"]:
        try:
            df[col] = df[col].str.replace(pat=r"\s", repl="")
        except AttributeError:
            pass
```

```python
df.to_csv("/Volumes/data/export_ddmmyyyy_white_space.csv")
```

## Final Validation Of Unique Values
We print once more the unique values for the columns in the DataFrame, to check
for any errors. No column shows signs of messy data anymore, that lets us create
new encoded columns with factorized categorical values.


```python
for n in df.columns:
    try:
        col_stats[n]["unique_after_text_processing"] = df[n].unique()
    except KeyError:
        pass
    except TypeError:
        pass

for col in df.columns:
    try:
        print(f'{col}:\n{col_stats[col]["unique_after_text_processing"]}\n\n')
    except KeyError:
        pass
```

    auxiliary_costs:
    [190.84 117.95 249.83 ...  89.59 197.96  59.93]


    lat:
    [53.52943934 53.58414267 53.60449188 ... 53.58693861 53.55758333
     53.64864505]


    lng:
    [10.15235715 10.06842725  9.86423116 ... 10.01625824 10.03986901
     10.03996646]


    parking:
    [nan 'Tiefgaragen-Stellplatz' 'Tiefgaragen-Stellplätze'
     'Duplex-Stellplatz' 'Außenstellplatz' 'Garage' 'Stellplatz' 'Carport'
     'Stellplätze' 'Parkhaus-Stellplatz' 'Parkhaus-Stellplätze' 'Garagen'
     'Außenstellplätze' 'Carports']


    date_listed:
    ['2018-12-03T00:00:00.000000000' '2018-12-02T00:00:00.000000000'
     '2018-11-30T00:00:00.000000000' '2018-11-29T00:00:00.000000000'
     '2018-11-28T00:00:00.000000000' '2018-11-27T00:00:00.000000000'
     '2018-11-26T00:00:00.000000000' '2018-11-25T00:00:00.000000000'
     '2018-11-24T00:00:00.000000000' '2018-11-23T00:00:00.000000000'
     '2018-11-22T00:00:00.000000000' '2018-11-21T00:00:00.000000000'
     '2018-11-20T00:00:00.000000000' '2018-11-19T00:00:00.000000000'
     '2018-11-18T00:00:00.000000000' '2018-11-17T00:00:00.000000000'
     '2018-11-16T00:00:00.000000000' '2018-11-15T00:00:00.000000000'
     '2018-11-14T00:00:00.000000000' '2018-11-13T00:00:00.000000000'
     '2018-11-12T00:00:00.000000000' '2018-11-11T00:00:00.000000000'
     '2018-03-03T00:00:00.000000000' '2017-02-12T00:00:00.000000000'
     '2016-11-27T00:00:00.000000000']


    total_energy_need:
    [nan '132.9' '65.9' '49' '50' '140' '158' '242.7' '23' '181.8' '176.6'
     '231.6' '102' '127' '57.2' '58.3' '148' '145' '126' '106.6' '189.4'
     '176.8' '138' '82.5' '110.3' '81' '90' '124' '68' '283' '257.8' '167.4'
     '157.9' '30.8' '66.3' '55' '85' '52.4' '27.5' '77' '169.9' '58.0' '65'
     '159' '73.4' '65.7' '129' '272.5' '55.2' '38.4' '69.8' '244.6' '148.2'
     '119.6' '39.7' '189.1' '143' '97' '87.1' '178' '224.6' '86.3' '64' '114'
     '89' '123' '57.5' '98' '218.3' '142' '48.8' '131' '117' '197' '92' '52'
     '581' '130' '128' '33' '54.8' '24.4' '211.4' '70.2' '134' '164' '200'
     '162' '156' '313' '53' '72' '275' '173' '116' '133' '95.4' '193.7'
     '183.9' '47.6' '204.4' '59.0' '225.3' '137.8' '102.7' '204.9' '34.5'
     '216.6' '69.6' '147.5' '94' '41.3' '100' '93' '121' '53.6' '70.7' '163'
    '270.7' '191.4' '189' '96.4' '238.8' '243.3' '289.4' '232.1' '249.8'
    '259']


    no_rooms:
    ['3' '2.5' '2' '1.5' '3.5' '1' '4' '4.5' '5' '6' '5.5' '6.5' '7.5' '8' '7'
     '36' '14']


    type:
    [nan 'Sonstige' 'Dachgeschoss' 'Etagenwohnung' 'Hochparterre'
     'Erdgeschosswohnung' 'Maisonette' 'Souterrain' 'Terrassenwohnung'
     'Penthouse' 'Loft']


    floor:
    ['0' '3' '2' nan '1' '6' '5' '4' '7' '14' '9' '10' '11' '8' '22' '-1' '19'
     '12' '13' '24' '15' '23' '99']


    no_bedrooms:
    [-9.999e+03  1.000e+00  2.000e+00  3.000e+00  4.000e+00  0.000e+00
      5.000e+00  6.000e+00]


    no_bathrooms:
    [-9.999e+03  1.000e+00  2.000e+00  0.000e+00  3.000e+00  1.100e+01]


    date_unlisted:
    ['2018-12-03T00:00:00.000000000' '2018-12-02T00:00:00.000000000'
     '2018-11-30T00:00:00.000000000' '2018-12-01T00:00:00.000000000'
     '2018-11-29T00:00:00.000000000' '2018-11-28T00:00:00.000000000'
     '2018-11-27T00:00:00.000000000' '2018-11-26T00:00:00.000000000'
     '2018-11-25T00:00:00.000000000' '2018-11-24T00:00:00.000000000'
     '2018-11-23T00:00:00.000000000' '2018-11-22T00:00:00.000000000'
     '2018-11-21T00:00:00.000000000' '2018-11-20T00:00:00.000000000'
     '2018-11-19T00:00:00.000000000' '2018-11-17T00:00:00.000000000'
     '2018-11-16T00:00:00.000000000' '2018-11-15T00:00:00.000000000'
     '2018-11-18T00:00:00.000000000' '2018-11-14T00:00:00.000000000'
     '2018-11-13T00:00:00.000000000' '2018-11-12T00:00:00.000000000'
     '2017-11-25T00:00:00.000000000' '2018-08-19T00:00:00.000000000'
     '2016-12-10T00:00:00.000000000']


    json_heating_type:
    [nan 'self_contained_central_heating' 'central_heating' 'no_information'
     'district_heating' 'electric_heating' 'night_storage_heater'
     'gas_heating' 'combined_heat_and_power_plant' 'floor_heating'
     'oil_heating' 'wood_pellet_heating' 'stove_heating' 'heat_pump']


    json_picturecount:
    ['6' '11' '5' '0' '13' '2' '4' '7' '3' '10' '1' '9' '8' '15' '12' '24'
     '17' '14' '21' '18' '19' '25' '20' '22' '40' '29' '16' '28' '27' '36'
     '35' '34' '26' '23' '43' '32' '31' '38' '30' '49' '39' '37' '42' '41'
     '45' '64']


    json_telekomdownloadspeed:
    ['100' '50' '16' nan '25' '6' '200']


    json_total_rent:
    ['541.06' '559.05' '839.01' ... '1189.24' '382.73' '499.91']


    json_yoc:
    ['1976' '1950' '1997' '1951' '2001' '1870' '1985' '1937' '1912' '1954'
     '1953' nan '1952' '1961' '2011' '1973' '1968' '1908' '1930' '1939' '1964'
     '1234' '1981' '1998' '1922' '1959' '1904' '1969' '1928' '2012' '2000'
     '1932' '2006' '2013' '1957' '1963' '1972' '1956' '1984' '1960' '1971'
     '1918' '1999' '1967' '1978' '1996' '1874' '1949' '1962' '1977' '1955'
     '1974' '1931' '1894' '1927' '2018' '1995' '1966' '1900' '1965' '1958'
     '1994' '2009' '1905' '1979' '1902' '1970' '1878' '2015' '1936' '2016'
     '1919' '1886' '2019' '1938' '1907' '1980' '1935' '2014' '1892' '1921'
     '2010' '1906' '1880' '1947' '2017' '1903' '1982' '1914' '1926' '1933'
     '1975' '1924' '1925' '1986' '2008' '1988' '1993' '1923' '2007' '2002'
     '1992' '1983' '1948' '1910' '1929' '1987' '1990' '1901' '1889' '1885'
     '1920' '1917' '1989' '1915' '1888' '1911' '1940' '1890' '1913' '1934'
     '1946' '2005' '1882' '1899' '1944' '2003' '2004' '1893' '1897' '1942'
     '1943' '1941' '1799' '1945' '1991' '1896' '1850' '1898' '1869' '1916'
     '1909' '1875' '1891' '1858' '1895' '1887' '1865' '1667']


    json_main_es:
    ['district_heating' 'no_information' 'oil' 'electricity' nan 'gas'
     'natural_gas_heavy' 'natural_gas_light' 'local_heating' 'geothermal'
     'liquid_gas' 'heat_supply' 'pellet_heating' 'solar_heating']


    json_bfitted_kitchen:
    ['y' 'n']


    json_cellar:
    ['n' 'y']


    json_const_time:
    ['3' '1' '5' '2' '6' '4' nan '8' '7' '9']


    json_base_rent:
    ['350.22' '441.1' '589.18' ... '991.28' '301.18' '398.91']


    json_square_meters:
    ['76' '60' '75' ... '63.99' '69.56' '46.93']


    json_object_condition:
    ['no_information' 'well_kept' 'fully_renovated' 'first_time_use'
     'first_time_use_after_refurbishment' 'mint_condition' 'modernized'
     'need_of_renovation' 'refurbished' 'negotiable']


    json_interiorqual:
    ['no_information' 'normal' 'sophisticated' 'simple' 'luxury']


    json_bpets_allowed:
    ['no_information' 'yes' 'negotiable' 'no']


    json_belevator:
    ['n' 'y']


    time_listed:
    [                0   259200000000000    86400000000000   172800000000000
       518400000000000   432000000000000   777600000000000   864000000000000
       345600000000000   691200000000000   950400000000000   604800000000000
      1123200000000000  1209600000000000  1036800000000000  1468800000000000
      1555200000000000  1296000000000000  1641600000000000  1814400000000000
      2073600000000000  1728000000000000  1382400000000000  1900800000000000
      1987200000000000  2246400000000000  2332800000000000  2592000000000000
      2505600000000000  2764800000000000  2937600000000000  2419200000000000
      3456000000000000  4060800000000000  2160000000000000  3024000000000000
      3283200000000000  4147200000000000  3974400000000000  3369600000000000
      3110400000000000  3888000000000000  3715200000000000  4924800000000000
    18835200000000000 39398400000000000 17971200000000000 11750400000000000
     15292800000000000 24019200000000000]





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9423 entries, 0 to 12323
    Data columns (total 30 columns):
     #   Column                     Non-Null Count  Dtype          
    ---  ------                     --------------  -----          
     0   auxiliary_costs            9423 non-null   float64        
     1   lat                        9423 non-null   float64        
     2   lng                        9423 non-null   float64        
     3   parking                    2065 non-null   object         
     4   date_listed                9423 non-null   datetime64[ns]
     5   total_energy_need          2796 non-null   object         
     6   no_rooms                   9423 non-null   object         
     7   type                       7089 non-null   object         
     8   floor                      7923 non-null   object         
     9   no_bedrooms                9423 non-null   float64        
     10  no_bathrooms               9423 non-null   float64        
     11  date_unlisted              9423 non-null   datetime64[ns]
     12  json_heating_type          7349 non-null   object         
     13  json_bbalcony              9423 non-null   object         
     14  json_picturecount          9423 non-null   object         
     15  json_telekomdownloadspeed  8959 non-null   object         
     16  json_total_rent            8844 non-null   object         
     17  json_yoc                   8688 non-null   object         
     18  json_main_es               8940 non-null   object         
     19  json_bfitted_kitchen       9423 non-null   object         
     20  json_cellar                9423 non-null   object         
     21  json_const_time            8688 non-null   object         
     22  json_base_rent             9423 non-null   object         
     23  json_square_meters         9073 non-null   object         
     24  json_object_condition      9423 non-null   object         
     25  json_interiorqual          9423 non-null   object         
     26  json_bpets_allowed         9423 non-null   object         
     27  json_belevator             9423 non-null   object         
     28  gps                        9423 non-null   object         
     29  time_listed                9423 non-null   timedelta64[ns]
    dtypes: datetime64[ns](2), float64(5), object(22), timedelta64[ns](1)
    memory usage: 2.5+ MB


We drop all non `json_` columns, that have a `json_` counterpart and replace
them by their respective `json_` counterparts.


```python
remove_prefix_name = []
json_name = []
for col in df.columns:
    if re.match(r"^json", col):
        json_name.append(col)
        pat = re.compile("^json_")
        remove_prefix_col = pat.sub("", col)
        remove_prefix_name.append(remove_prefix_col)
        print(remove_prefix_col)

print(remove_prefix_name, len(remove_prefix_name))
```

    heating_type
    bbalcony
    picturecount
    telekomdownloadspeed
    total_rent
    yoc
    main_es
    bfitted_kitchen
    cellar
    const_time
    base_rent
    square_meters
    object_condition
    interiorqual
    bpets_allowed
    belevator
    ['heating_type', 'bbalcony', 'picturecount', 'telekomdownloadspeed', 'total_rent', 'yoc', 'main_es', 'bfitted_kitchen', 'cellar', 'const_time', 'base_rent', 'square_meters', 'object_condition', 'interiorqual', 'bpets_allowed', 'belevator'] 16



```python
df.columns
```




    Index(['auxiliary_costs', 'lat', 'lng', 'parking', 'date_listed',
           'total_energy_need', 'no_rooms', 'type', 'floor', 'no_bedrooms',
           'no_bathrooms', 'date_unlisted', 'json_heating_type', 'json_bbalcony',
           'json_picturecount', 'json_telekomdownloadspeed', 'json_total_rent',
           'json_yoc', 'json_main_es', 'json_bfitted_kitchen', 'json_cellar',
           'json_const_time', 'json_base_rent', 'json_square_meters',
           'json_object_condition', 'json_interiorqual', 'json_bpets_allowed',
           'json_belevator', 'gps', 'time_listed'],
          dtype='object')




```python
for col in zip(json_name, remove_prefix_name):
    df = df.rename_column(col[0], col[1])
```


```python
for col in zip(json_name, remove_prefix_name):
    print(col)
```

    ('json_heating_type', 'heating_type')
    ('json_bbalcony', 'bbalcony')
    ('json_picturecount', 'picturecount')
    ('json_telekomdownloadspeed', 'telekomdownloadspeed')
    ('json_total_rent', 'total_rent')
    ('json_yoc', 'yoc')
    ('json_main_es', 'main_es')
    ('json_bfitted_kitchen', 'bfitted_kitchen')
    ('json_cellar', 'cellar')
    ('json_const_time', 'const_time')
    ('json_base_rent', 'base_rent')
    ('json_square_meters', 'square_meters')
    ('json_object_condition', 'object_condition')
    ('json_interiorqual', 'interiorqual')
    ('json_bpets_allowed', 'bpets_allowed')
    ('json_belevator', 'belevator')



```python
df.columns
```




    Index(['auxiliary_costs', 'lat', 'lng', 'parking', 'date_listed',
           'total_energy_need', 'no_rooms', 'type', 'floor', 'no_bedrooms',
           'no_bathrooms', 'date_unlisted', 'heating_type', 'bbalcony',
           'picturecount', 'telekomdownloadspeed', 'total_rent', 'yoc', 'main_es',
           'bfitted_kitchen', 'cellar', 'const_time', 'base_rent', 'square_meters',
           'object_condition', 'interiorqual', 'bpets_allowed', 'belevator', 'gps',
           'time_listed'],
          dtype='object')



## Reordering Of Columns

We reorder the columns with dtype `categorical` to be at the beginning of the
columns in `df.columns`, to make sub-setting of the columns during the *feature
selection* and *model evaluation* phases. The *NaN*
values in the dtype `categorical` columns are replaced with `no_information`.
The reason being, that rows with recognized *NaN* values get excluded from the
factorization process and are replaced by `-1`. At this point, we like to keep
the option of using the missing values as a categorical group of its own. The
group, that contains all listings where no information is given for.


```python
df = df.reorder_columns(
        column_order=[
                "bbalcony",
                "bfitted_kitchen",
                "cellar",
                "belevator",
                "parking",
                "type",
                "floor",
                "no_rooms",
                "no_bedrooms",
                "no_bathrooms",
                "heating_type",
                "main_es",
                "const_time",
                "object_condition",
                "interiorqual",
                "bpets_allowed",
                "gps",
                "time_listed",
                "auxiliary_costs",
                "total_energy_need",
                "picturecount",
                "total_rent",
                "yoc",
                "base_rent",
                "square_meters",
                ]
        ).fill_empty(
        column_names=[
                "parking",
                "type",
                "heating_type",
                "object_condition",
                "bpets_allowed",
                "interiorqual",
                "main_es",
                ],
        value="no_information",
        )
```


```python
category_cols = [
        "bbalcony",
        "bfitted_kitchen",
        "cellar",
        "belevator",
        "parking",
        "type",
        "heating_type",
        "main_es",
        "object_condition",
        "interiorqual",
        "bpets_allowed",
        ]
```


```python
df = df.factorize_columns(column_names=category_cols).encode_categorical(
        column_names=[f"{col}_enc" for col in category_cols]
        )
```

## Summary Statistics Of The Categorical Columns


```python
df.filter(regex=r'_enc').info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9423 entries, 0 to 12323
    Data columns (total 11 columns):
     #   Column                Non-Null Count  Dtype   
    ---  ------                --------------  -----   
     0   bbalcony_enc          9423 non-null   category
     1   bfitted_kitchen_enc   9423 non-null   category
     2   cellar_enc            9423 non-null   category
     3   belevator_enc         9423 non-null   category
     4   parking_enc           9423 non-null   category
     5   type_enc              9423 non-null   category
     6   heating_type_enc      9423 non-null   category
     7   main_es_enc           9423 non-null   category
     8   object_condition_enc  9423 non-null   category
     9   interiorqual_enc      9423 non-null   category
     10  bpets_allowed_enc     9423 non-null   category
    dtypes: category(11)
    memory usage: 436.5 KB


## The Next Steps
This marks the end of the initial cleaning of the tabular data from the web
scraping part. The next steps are:
- Create a machine learning pipeline with reproducible steps. A pipeline, that includes removal of missing values or imputation of missing values among other data preprocessing steps that might be needed.
- Create stratified *k*-fold cross validation splits and a train and test set.
- Try different scaling algorithms after the split on the independent variables and feature elimination techniques.
- Decide whether prediction accuracy is the key metric for this project or model interpretability.
- Evaluate the univariate, bivariate and multivariate distributions of the columns as part of choosing candidate models and evaluating their predictions for the value of the dependent variable `base_rent`.
- Find the key columns for each candidate model, the ones that are relevant to the model's performance. Remove the others.
- Perform Feature Creation using independent geo-spacial data and linking it to the listings. Test, which features are not relevant for the model's performance.

We export the DataFrame in its current state for further work.

```python
df.to_csv("/Volumes/data/df_first_cleaning.csv")

```

---
<br>
<br>
[Data Preparation Series 1]({% link _projects/data_prep_1.md %})  
[Data Preparation Series 2]({% link _projects/data_prep_2.md %})  
[Data Preparation Series 3]({% link _projects/data_prep_3.md %})  
[Data Preparation Series 4]({% link _projects/data_prep_4.md %})  
