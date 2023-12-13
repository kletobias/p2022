## Header Part 1 of the series

```json
{
    "Mastery in Pandas: In-Depth Data Exploration, Part 1": {
        "status": "updated",
        "description": "This article showcases my expertise in using pandas for advanced data exploration. It focuses on analyzing a 47-column dataset, providing insights into leveraging pandas for complex tabular data management.",
        "tags": [
            "data-exploration",
            "advanced-pandas",
            "data-analysis",
            "tabular-data"
        ],
        "category": "data-preprocessing",
        "word_count": 3508
},
    "PyJanitor Proficiency: Efficient String Data Cleaning, Part 2": {
        "status": "updated",
        "description": "Demonstrates my proficiency with pyjanitor in streamlining string data cleaning. This installment details method chaining and advanced string manipulation techniques for optimizing data preprocessing.",
        "tags": ['data-cleaning', 'pyjanitor', 'string-manipulation', 'method-chaining', 'pandas'],
        "category": "data-preprocessing",
        "word_count": 3265,
        "full_article": "https://deep-learning-mastery.com/projects/pyjanitor-proficiency-efficient-string-data-cleaning-part-2/"
    },
    "Geospatial Engineering in Pandas: Creating Valid GPS Columns, Part 3": {
        "status": "todo",
        "description": "Highlights my skills in geospatial data engineering using pandas. This part focuses on cleaning, validating, and creating GPS data columns by merging longitude and latitude fields into Point geometry objects.",
        "tags": [
            "geospatial-data",
            "data-validation",
            "GPS-data-creation",
            "pandas",
            "geometry-manipulation"
        ],
        "category": "data-preprocessing",
        "word_count": 2372,
        "full_article": "https://deep-learning-mastery.com/projects/geospatial-engineering-in-pandas-creating-valid-gps-columns-part-3/"
    },
    "Advanced Data Cleaning and Validation: Batch Processing with Pandas, Part 4": {
        "status": "todo",
        "description": "Showcases advanced techniques in data cleaning and validation, using batch processing with pandas and regular expressions. It exemplifies how to handle large volumes of tabular data efficiently.",
        "tags": [
            "advanced-data-cleaning",
            "batch-processing",
            "pandas",
            "regular-expressions",
            "data-validation"
        ],
        "category": "data-preprocessing",
        "word_count": 4259,
        "full_article": "https://deep-learning-mastery.com/projects/advanced-data-cleaning-and-validation-batch-processing-with-pandas-part-4/"
    }
}
```

Rewrite the following summary given the above inputs from the four files that are part of the series:


## Summary Of The Series

> A DataFrame is given as input that contains 47 columns at the beginning.
  Dimensionality Reduction is performed on the columns, to filter and only keep relevant columns.
  The `pyjanitor` module is widely used with its *method chaining syntax* to increase the speed of the cleaning procedure.
  Unique values of each column give the basis for the steps needed to clean the columns.
  Regular Expressions (**regex**) are mostly used to extract cell contents that hold the valid data.
  Regex are also used to replace invalid character patterns with valid ones.
  Validation of the values, after cleaning is performed using regex patterns.
  New `timedelta64[ns]` `time_listed` and `Point` geometry `gps` columns are created.
