---
layout: distill
title: 'How To Import CSV Files Into MySQL Using Python And Pandas.'
date: 2022-04-01 06:32:00
description: 'There is no functionality built into MySQL that lets one import CSV data directly into a database, in the form of a table. pandas can be used to import CSV files from within Python, into a table of a database.'
tags: ['import csv data','mysql','mysql-connector-python','pandas','sqlalchemy']
category: 'MySQL'
comments: true
---

# Using Pandas To Import CSV Data Into MySQL

Using pandas, one can import any data type that pandas can read and create a DataFrame
object from. The full list of possible input file types can be found here: [IO tools (text, CSV, HDF5, …) — pandas documentation](https://pandas.pydata.org/docs/user_guide/io.html#io-tools-text-csv-hdf5).  
In the following, only **CSV** data will be mentioned, but everything applies to all the file types mentioned in the link above, once a `pandas.DataFrame` object has been created.

## What Problem Does The Solution Solve?

There is no functionality built into **MySQL** that lets one import **CSV** data directly into a database, in the form of a table. pandas can be used to import CSV files from within Python, into a table of a database.

## Prerequisites

The following Python packages are needed and can be installed using **pip** for the import through the pandas module.  

In the case of having a Python **virtual environment** ([venv](https://docs.python.org/3/library/venv.html#module-venv) in this case), first activate the virtual environment and then run the following commands, like shown below.

```shell
# activating the virtual environment 
source venv/bin/activate
```

```shell
# installing the Python modules needed
pip install SQLAlchemy
pip install pandas
pip install mysql-connector-python
```

## Importing The Data

The following is a step-by-step walk through of how the end-to-end import of any CSV file into a MySQL database can
look like. It is assumed that the database is hosted locally and `localhost` is used as hostname. It can be replaced by
the URL of the database.

### Import Of The Python Packages
```python
# Import the modules needed
import pandas as pd
from sqlalchemy import create_engine
# importing what is needed from mysql-connector-python module
import mysql.connector
```


### Create The Database Engine

```python
engine = create_engine(
"mysql+mysqlconnector://{user}:{pw}@localhost/{db}"
.format(
user="username",
pw="password",
db="database/schema"
)
)
print("Done creating the engine")
```

One creates the engine using `create_engine()` from the `sqlalchemy` package and fills in the values for 'user', 'pw' and 'db'. 'db' is optional and can be omitted.



### Set The Data Types Of The Columns

For that, the needed types from the `sqlalchemy` package are imported. The
entire  list of data types supported by `sqlalchemy` can be found
[here](https://docs.sqlalchemy.org/en/14/core/type_basics.html#generic-types).

```python
from sqlalchemy.types import Integer, Float, Text, DateTime
```

### The Actual Import

In this instance, parts of the yellow cab dataset from the NYC Taxi dataset were
used. There were are total of ~44.000.000 rows to be imported by the process
below.

```python
df = pd.read_csv("path to the .csv file")
print('Done creating the pandas DataFrame object, it has %d number of rows' % len(df))
df.to_sql(
"yp_01_06",
con = engine,
if_exists = "replace",
#	schema = "datasets", uncomment if no database specified in the engine
	index = False,
	chunksize = 1000,
	dtype={
		"VendorID": Integer,
		"tpep_pickup_datetime": DateTime,
		"tpep_dropoff_datetime": DateTime,
		"passenger_count": Integer,
		"trip_distance": Float,
		"RatecodeID": Integer,
		"store_and_fwd_flag": Text,
		"PULocationID": Integer,
		"DOLocationID": Integer,
		"payment_type": Integer,
		"fare_amount": Float,
		"extra": Float,
		"mta_tax": Float,
		"tip_amount": Float,
		"tolls_amount": Float,
		"improvement_surcharge": Float,
		"total_amount": Float,
		"congestion_surcharge": Text
		}
)
print("Done")
```

There are several important parameters to be specified, which will be highlighted in  
the following.

- `con` is the engine created earlier.
- `if_exists` can either be `"replace"` or `"append"`. Append will cause the data in the `pandas.DataFrame` to be appended to the already existing data, in case there is already data in the table.
- `chunksize` must not be larger 1000, otherwise MySQL may throw an error during the import.
- `dtypes` is a dictionary in which the data types for all the columns of the `pandas.DataFrame` are specified. These type assignments must be correct or errors are likely to be thrown during the import process by MySQL.

The process of importing the data can take anywhere from seconds to minutes, depending on the number of columns and rows to be imported, among other factors.

I am glad, if the article was able to help you in solving the challenge of how to import CSV data into MySQL. Python is a versatile tool and `sqlalchemy` makes it possible to use all the tools that MySQL has from within the Python universe. With the respective driver, it also works for PostgreSQL, Oracle, Microsoft Server, SQLite among others.
