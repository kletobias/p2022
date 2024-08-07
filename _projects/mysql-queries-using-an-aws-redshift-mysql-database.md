---
layout: distill
title: 'Integrating MySQL with AWS Redshift for Data Analysis'
date: 2023-02-06
description: 'Demonstrates the integration of Python, Pandas, and MySQL in an AWS Redshift environment, guiding through the process of data import and formulation of MySQL queries for effective data retrieval.'
img: 'assets/img/838338477938@+-98398438.jpg'
tags: ['AWS-redshift', 'mysql-integration', 'python-data-import', 'data-querying', 'pandas-usage']
category: ['Tabular Data']
authors: 'Tobias Klein'
comments: true
---
<br>
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#summary">Summary</a></div>
    <div class="no-math"><a href="#import-using-sqlalchemy">Import Using sqlalchemy</a></div>
    <div class="no-math"><a href="#source-code-of-the-script">Source Code Of The Script</a></div>
    <div class="no-math"><a href="#sending-mysql-queries-to-the-database">Sending MySQL Queries To The Database</a></div>
    <div class="no-math"><a href="#conclusion">Conclusion</a></div>
  </nav>
</d-contents>

# Integrating MySQL with AWS Redshift for Data Analysis

## Summary
This article demonstrates the process of bulk importing flat files and other files readable by the `Pandas` Python library into a MySQL database. It also covers sending basic MySQL queries to an AWS Redshift hosted MySQL database. The featured dataset is the 'Video Game Sales' dataset, available on Kaggle at: [*Video Games Dataset*](https://www.kaggle.com/datasets/gregorut/videogamesales).

## Import Using SQLAlchemy
The import method is applicable to any database compatible with the `sqlalchemy` library and supported by a Python *connector* library. MySQL provides a native tool for importing text files into a database: [**mysqlimport**](https://dev.mysql.com/doc/refman/8.0/en/mysqlimport.html).

### create_engine Function
At the heart of this approach is the `create_engine` function from `sqlalchemy`, complemented by `sqlalchemy.types`, which determines the destination data types for converting the input data.

### MySQL Connector & Pandas
`mysql.connector` is employed behind the scenes to establish a connection with the target schema and table. The `.to_sql()` method on a DataFrame facilitates the data import process. Standard libraries `os` and `re` are utilized for file system access, filtering, and navigation.

### AWS Redshift Connection Syntax
Connecting to a Redshift database instance involves setting up specific parameters and inputting the correct values. Essential steps include:

#### Endpoint
The **Endpoint** in AWS terminology is the **host** component of the connection URL, typically resembling:

```python
host='database.nggdttw.eu-central.rds.amazonaws.com'
```

#### User & Password

Creation of the database requires specifying **User** or users and a corresponding **password**.

#### Database Public Availability

By default, the database is inaccessible from outside its AWS VPC group. To enable external access, one must set 'publicly available' to true. Then, within the specified VPC security group, add new rules if necessary. Under 'inbound' rules, define the range of IP addresses permitted to connect to the database. Options include using 'Current IP Address' for allowing only the current machine's public IP or '0.0.0.0' for unrestricted access. The default 'outbound' rules usually suffice. With these settings in place, the `engine` creation as shown below should successfully establish a connection.

```python
u = 'root' # for the case of local mysql instance running on UNIX socket.
pw= '' # no password by default for local mysql instance for user 'root'
#input_scheme = input()
host='database.nggdttw.eu-central.rds.amazonaws.com' # localhost for UNIX socket
def ce(host,u=u,pw=pw,scheme='MAIN1'):
	engine = create_engine(
			f"mysql+mysqlconnector://{u}:{pw}@{host}/{scheme}"
					)
	try:
		print(f'Done creating the engine for: mysql+mysqlconnector://{u}:{pw}@{host}/{scheme}')
		return engine
	except Exception:
		print(f'Error creating the engine for: mysql+mysqlconnector://{u}:{pw}@{host}/{scheme}')

engine = ce(host=host,u='admin',pw='-------',scheme='PRAC1')
```


## Source Code Of The Script

```python
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import *
import os
import re

print('please input the value for scheme (synonymous for database')

u = 'root' # for the case of local mysql instance running on UNIX socket.
pw= '' # no password by default for local mysql instance for user 'root'
#input_scheme = input()
host='database.nggdttw.eu-central.rds.amazonaws.com' # localhost for UNIX socket

def ce(host,u=u,pw=pw,scheme='MAIN1'):
	engine = create_engine(
			f"mysql+mysqlconnector://{u}:{pw}@{host}/{scheme}"
					)
	try:
		print(f'Done creating the engine for: mysql+mysqlconnector://{u}:{pw}@{host}/{scheme}')
		return engine
	except Exception:
		print(f'Error creating the engine for: mysql+mysqlconnector://{u}:{pw}@{host}/{scheme}')

engine = ce(host=host,u='admin',pw='-------',scheme='PRAC1')

print('please input the value for data_dir as an absolute path\nwithout trailing slash')
data_dir='/Volumes/data/dataset-practice/video-game-sales'

maptype = {"dtype('O')":"Text","dtype('int64')":"Integer","dtype('float64')":"Float"}

def get_convert(data_dir=data_dir,engine=engine,maptype=maptype):
	files=os.listdir(data_dir)
	pat = re.compile('\.csv$',re.IGNORECASE)
	csv_files=[file for file in files if pat.search(file)]
	maptype = {"dtype('O')":"Text","dtype('int64')":"Integer","dtype('float64')":"Float"}
	for f in csv_files:
		l = data_dir + "/"+ f
		df=pd.read_csv(l,low_memory=False)
		print(f'done creating df for file {f}')
		dtype = dict(zip(df.columns.tolist(),list()))
		dtypesf = [df[col].dtype for col in df.columns]
		for i in df.columns.tolist():
			print(i)
		for j in dtypesf:
			print(j)
		try:
			for (col,dtypecol) in zip(df.columns.tolist(),dtypesf):
				for i in maptype.keys():
					if dtypecol == str(i[1]):
						dtype[i[0]].append(maptype[dtypecol])
		except ValueError:
			print('problem with "maptype" line ~60')




				
					
		df.to_sql(
                name=f[:-4],
				con=engine,
				if_exists='replace',
				index=False,
				chunksize=1000,
				dtype = dtype,
				)
	print('done')

get_convert()
```

## Sending MySQL Queries To The Database
With the *csv* file imported into the database, one can begin sending queries to
the MySQL database instance hosted on **AWS** using *Redshift* (**RDS**).

**Notice: All queries that return a table only have the first five rows printed
out at maximum for readability.**

### CREATE SCHEMA
If there are no `SCHEMAS` in the database, create schema `PRAC1`.

```sql
CREATE SCHEMA IF NOT EXISTS PRAC1;
```

### SHOW SCHEMAS
Get a list of all `SCHEMAS` in database `database-1`.

```sql
SHOW SCHEMAS;
```

```
+--------------------+
| Database           |
+--------------------+
| Chinook            |
| PRAC1              |
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
```

### SHOW TABLES
Get a list of all tables in scheme/database `PRAC1`.

```sql
USE PRAC1;
SHOW TABLES;
```

```
+-----------------+
| Tables_in_PRAC1 |
+-----------------+
| vgs             |
+-----------------+
```

### DESCRIBE TABLE
Characteristics of table `vgs` are printed out.

```sql
USE PRAC1;
DESCRIBE TABLE vgs;
```

```
+----+-------------+-------+------------+------+---------------+------+---------+------+-------+----------+-------+
| id | select_type | table | partitions | type | possible_keys | key  | key_len | ref  | rows  | filtered | Extra |
+----+-------------+-------+------------+------+---------------+------+---------+------+-------+----------+-------+
|  1 | SIMPLE      | vgs   | NULL       | ALL  | NULL          | NULL | NULL    | NULL | 15996 |   100.00 | NULL  |
+----+-------------+-------+------------+------+---------------+------+---------+------+-------+----------+-------+
```

### SHOW COLUMNS
One can get the column names of the table using `SHOW COLUMNS`.

```sql
USE PRAC1;
SHOW COLUMNS
FROM vgs;
```

```
+--------------+--------+------+-----+---------+-------+
| Field        | Type   | Null | Key | Default | Extra |
+--------------+--------+------+-----+---------+-------+
| Rank         | bigint | YES  |     | NULL    |       |
| Name         | text   | YES  |     | NULL    |       |
| Platform     | text   | YES  |     | NULL    |       |
| Year         | double | YES  |     | NULL    |       |
| Genre        | text   | YES  |     | NULL    |       |
| Publisher    | text   | YES  |     | NULL    |       |
| NA_Sales     | double | YES  |     | NULL    |       |
| EU_Sales     | double | YES  |     | NULL    |       |
| JP_Sales     | double | YES  |     | NULL    |       |
| Other_Sales  | double | YES  |     | NULL    |       |
| Global_Sales | double | YES  |     | NULL    |       |
+--------------+--------+------+-----+---------+-------+
```



### SELECT Statement
Columns `Rank`, `Publisher`, `Global_Sales` are selected from table `vgs` and
the first five rows are printed out. It is like the 'head' function.

```sql
USE PRAC1;
SELECT `Rank`, `Publisher`, `Global_Sales` 
FROM vgs
LIMIT 5;
```

```
+------+-----------+--------------+
| Rank | Publisher | Global_Sales |
+------+-----------+--------------+
|    1 | Nintendo  |        82.74 |
|    2 | Nintendo  |        40.24 |
|    3 | Nintendo  |        35.82 |
|    4 | Nintendo  |           33 |
|    5 | Nintendo  |        31.37 |
+------+-----------+--------------+
```

### WHERE LIKE Condition
Columns `Rank`, `Year`, `Publisher`, `Global_Sales` are selected and the rows
printed out where `Publisher` contains the equivalent *Glob* pattern
"\*Valve\*".

```sql
USE PRAC1;
SELECT `Rank`, `Year`, `Publisher`, `Global_Sales` 
FROM vgs
WHERE `Publisher` LIKE "%Valve%"
ORDER BY `Global_Sales` DESC
LIMIT 5;
```

```
+------+------+----------------+--------------+
| Rank | Year | Publisher      | Global_Sales |
+------+------+----------------+--------------+
|  791 | 2011 | Valve Software |          2.1 |
| 1018 | 2011 | Valve          |         1.74 |
| 2701 | 2011 | Valve Software |         0.76 |
| 5160 | 2009 | Valve Software |         0.37 |
+------+------+----------------+--------------+
```


### LEFT JOIN
We get the `Name`, `Publisher`, `NA_Sales`, `EU_Sales` columns from the 'left'
table of the inner join with itself, where the `EU_Sales` are larger than the
`NA_Sales` of the matching row of the right side table and print out the first
five rows with the most right column being the `Name` of the matching game
title. We group by unique `Publisher` from the left side table and sort in
descending order by left side table `Global_Sales`.

```sql
USE PRAC1;
SELECT v1.Name, v1.Publisher, v1.NA_Sales, v1.EU_Sales, v2.Name
FROM vgs v1 
LEFT JOIN vgs v2
ON v1.EU_Sales > v2.NA_Sales
GROUP BY v1.Publisher
ORDER BY v1.`Global_Sales` DESC
LIMIT 5;
```

```
+--------------------------------+-----------------------------+----------+----------+--------------------------+
| Name                           | Publisher                   | NA_Sales | EU_Sales | Name                     |
+--------------------------------+-----------------------------+----------+----------+--------------------------+
| Wii Sports                     | Nintendo                    |    41.49 |    29.02 | Tony Hawk's Downhill Jam |
| Kinect Adventures!             | Microsoft Game Studios      |    14.97 |     4.94 | Tony Hawk's Downhill Jam |
| Grand Theft Auto V             | Take-Two Interactive        |     7.01 |     9.27 | Tony Hawk's Downhill Jam |
| Gran Turismo 3: A-Spec         | Sony Computer Entertainment |     6.85 |     5.09 | Tony Hawk's Downhill Jam |
| Call of Duty: Modern Warfare 3 | Activision                  |     9.03 |     4.28 | Tony Hawk's Downhill Jam |
+--------------------------------+-----------------------------+----------+----------+--------------------------+
```

### INNER JOIN
This query selects columns `v1.Name`, `v1.NA_Sales`, `v2.Global_Sales`,
`v2.EU_Sales`, `v2.Name` from the same table vgs using aliases `v1` and `v2` for
the inner join. Rows returned are ones, where `v2.EU_Sales` is larger
`v1.NA_Sales` and in order to remove results of games that might not have been
for sale in NA, and therefore have a value of 0 in the `v1.NA_Sales` column, a
second condition is added that filters rows with a value of zero in the
`v1.NA_Sales` column. Results are then ordered by `v2.Global_Sales`.



```sql
USE PRAC1;
SELECT v1.Name, v1.NA_Sales, v2.Global_Sales, v2.EU_Sales, v2.Name
FROM vgs v1 
INNER JOIN vgs v2
ON v2.EU_Sales > v1.NA_Sales
AND v1.NA_Sales > 0
ORDER BY v2.Global_Sales
LIMIT 5;
```

```
+------------------+----------+--------------+----------+----------------------------------------+
| Name             | NA_Sales | Global_Sales | EU_Sales | Name                                   |
+------------------+----------+--------------+----------+----------------------------------------+
| Spirits & Spells |     0.01 |         0.02 |     0.02 | Jewel Link: Galactic Quest             |
| Spirits & Spells |     0.01 |         0.02 |     0.02 | Bella Sara 2 - The Magic of Drasilmare |
| Spirits & Spells |     0.01 |         0.02 |     0.02 | Pippa Funnell: Ranch Rescue            |
| Spirits & Spells |     0.01 |         0.02 |     0.02 | Demolition Company: Gold Edition       |
| Spirits & Spells |     0.01 |         0.02 |     0.02 | Ridge Racer Unbounded                  |
+------------------+----------+--------------+----------+----------------------------------------+
```

### Arithmetics And Alias 1
There are several [*Arithmetic
operators*](https://dev.mysql.com/doc/refman/8.0/en/arithmetic-functions.html)
in MySQL one can use. This is an example of calculating the difference between
`NA_Sales` and `EU_Sales`, renaming the resulting column with the `AS` operator
to `"SD"`. Using MySQL operator `WHERE` to add the condition that we only want
rows returned where `NA_Sales` is smaller `EU_Sales`. The results are grouped
by unique `Platform` value.

```sql
USE PRAC1;
SELECT `Platform`, `NA_Sales`-`EU_Sales` AS "SD"
FROM vgs
WHERE `NA_Sales` < `EU_Sales`
GROUP BY `Platform`
ORDER BY `Global_Sales` DESC
LIMIT 5;
```

```
+----------+-----------------------+
| Platform | SD                    |
+----------+-----------------------+
| DS       |   -1.9299999999999997 |
| PS3      |                 -2.26 |
| PS4      | -0.040000000000000036 |
| 3DS      |  -0.31000000000000005 |
| Wii      |   -1.7399999999999998 |
+----------+-----------------------+
```

### Arithmetics And Alias 2
Values in the 'Sales' columns are given in Millions. This example shows how one
can calculate the sum of all values in the `Global_Sales` column and transform
the resulting value into a number representing the total in Billions of sales.

```sql
USE PRAC1;
SELECT SUM(`Global_Sales`)/100 AS 'Total Billions'
FROM vgs;
```

```
+-------------------+
| Total Billions    |
+-------------------+
| 89.20440000001282 |
+-------------------+
```

## Conclusion
This article highlighted a way that one can use `Python` along some of it's
powerful libraries for data access and manipulation to import any flat file,
that pandas can read into a MySQL database instance. sqlalchemy is capable of
far more than just creating and mapping data types to match the ones of a
MySQL database. It can be used to import data straight from a database instance
into a pandas `DataFrame` for example. The queries one can send using sqlalchemy
are similar to the native MySQL queries with a few differences. This will be
explored in another article.

Thank you very much for reading this article. Please feel free to link to this
article or write a comment in the comments section below.

