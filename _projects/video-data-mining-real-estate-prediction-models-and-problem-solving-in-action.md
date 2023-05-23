---
layout: distill
title: 'Video Data Mining: Real-Estate Prediction Models and Problem-Solving in Action'
date: 2023-05-23
description: 'In the current era of data-driven decision-making, understanding the intricacies of data extraction, cleansing, validation, and enrichment is a valuable asset for businesses across all sectors.'
img: 'assets/img/838338477938@+-791693336-video.webp'
tags: ['etl', 'data-extraction', 'problem-solving', 'video', 'feature-enrichment']
category: ['tabular-data']
authors: 'Tobias Klein'
comments: true
featured: true
---


<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#summary">Summary</a></div>
    <div class="no-math"><a href="#video">Video</a></div>
    <div class="no-math"><a href="#related">Related</a></div>
  </nav>
</d-contents>

# Introduction to Data Extraction and Problem-Solving in Real Estate Analytics

## Summary

In the current era of <strong>data-driven decision-making</strong>,
understanding the intricacies of <i>data extraction, cleansing, validation, and
enrichment</i> is a valuable asset for businesses across all sectors.


> In this YouTube video, I showcase an intriguing project I completed: a deep-dive
into data extraction and pre-processing for a real-estate prediction model. 

This video underscores the nuances of working with <i>complex and unstructured
data, particularly from websites that don't readily offer <strong>API</strong>
access</i>. It mirrors the real-world scenario where important data can be
hidden or nested within unconventional formats, demanding innovative
problem-solving skills and a deep understanding of data structures. 

In my work, I demonstrate how I accessed an archive of rental apartment
listings, using these data to build an impactful machine learning model for real
estate pricing. This model was developed in the context of <strong>hyperparameter
optimization</strong> for `Lasso Regression` and `XGBoostRegressor` models, with <strong>predictive
performance</strong> and <strong>interpretability</strong> forming the critical evaluation criteria.

Beyond simply extracting and modeling the data, I emphasize the importance of
data cleansing, validation, and enrichment, which ultimately contribute to the
quality and accuracy of the resulting model. Notably, I delve into the problem
of accurately obtaining and validating GPS coordinates from the listings, and
how this information can be leveraged to provide additional value, such as
determining neighborhood noise levels or the distance to the nearest public
transportation station.

In an industry that's ever more reliant on data, this video provides a clear
demonstration of how meticulous data handling and innovative problem-solving can
yield profound insights. It serves as a testament to the applicability of these
methods to real-world industry problems, from improving the accuracy of
predictive models to enabling smarter, data-driven decision-making processes. 

Whether you are a data enthusiast, a professional working in real estate
analytics, or simply curious about how data science is applied in a real-world
context, this video has something for you. I invite you to watch, learn, and
discover how data, when correctly handled and interpreted, can indeed be
transformed into actionable insights.

## Video


<iframe width="660" height="415"
src="https://www.youtube-nocookie.com/embed/95_aIW4-F2s" title="YouTube video
player" frameborder="0" allow="accelerometer; autoplay; clipboard-write;
encrypted-media; gyroscope; picture-in-picture; web-share"
allowfullscreen></iframe>
<br>

## Related 

This article precedes the following articles.

### Cleaning a web scraped 47 Column Pandas DataFrame Part 1

- **Description:** Data Preparation Series: Exploring Tabular Data With pandas: An Overview Of Available Tools In The pandas Library.
- **Tags:** ['data-exploration', 'first-steps', 'introduction', 'pandas', 'tabular-data']
- **Category:** _data-preprocessing_ | **Word Count:** 3508 | **[Full Article](https://deep-learning-mastery.com/projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-1/)**

### Cleaning a web scraped 47 Column Pandas DataFrame Part 2

- **Description:** More efficient string data cleaning by using the pyjanitor module and method chaining.
- **Tags:** ['data-cleaning', 'pandas', 'regular-expressions', 'string-manipulation', 'tabular-data']
- **Category:** _data-preprocessing_ | **Word Count:** 3265 | **[Full Article](https://deep-learning-mastery.com/projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-2/)**

### Cleaning a web scraped 47 Column Pandas DataFrame Part 3

- **Description:** Extensive cleaning and validation and creation of a valid GPS column from the records, by joining the longitude and latitude columns together using geometry object Point.
- **Tags:** ['data-validation', 'dtype-timedelta64','geospatial-feature-engineering', 'pandas', 'tabular-data']
- **Category:** _data-preprocessing_ | **Word Count:** 2372 | **[Full Article](https://deep-learning-mastery.com/projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-3/)**

### Cleaning a web scraped 47 Column Pandas DataFrame Part 4

- **Description:** Extensive data cleaning and validation using regular expressions. Showcase of how batch processing several columns of tabular data using pandas, pyjanitor and the re library can look like.
- **Tags:** ['batch-processing', 'data-validation', 'pandas', 'regular-expressions', 'tabular-data']
- **Category:** _data-preprocessing_ | **Word Count:** 4259 | **[Full Article](https://deep-learning-mastery.com/projects/cleaning-a-web-scraped-47-column-br-pandas-dataframe-br-part-4/)**

All the articles mentioned here are parts of my Bachelor's thesis:

> **Data Mining: Hyperparameter Optimization For Real Estate Prediction Models.**

- [**Full Text (pdf)**]({% link assets/pdf/hyperparameter-optimization-bachelor-thesis-tobias-klein.pdf%})


<footer>
    <h2>Join the Conversation</h2>
    <p>Thank you for reading my article! Your opinions, ideas and questions are what keep this discussion alive and enriching. I'd love to hear what you think about this topic.</p>
    <p>If you found this piece interesting or insightful, please consider liking, sharing or leaving a comment below. Your interaction can help bring this information to more people and spark thoughtful discussion.</p>
    <p>Also, don't forget to follow me on social media for updates on new articles and my work. You can find the links to my profiles at the top or bottom of the page. I look forward to connecting with you!</p>
    <p>If you have any specific inquiries or opportunities for collaboration, feel free to reach out to me directly. I'm always open to new ideas and perspectives!</p>
</footer>

