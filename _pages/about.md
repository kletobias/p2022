---
layout: about
title: Home
permalink: /
#subtitle: <a href='#'>Affiliations</a>. Address. Contacts. Moto. Etc.
horizontal: false

profile:
  align: right
  image: profile.jpeg
  image_cicular: false # crops the image to make it circular
news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false  # includes social icons at the bottom of the page

hero_area:
  intro: "Hi I'm Tobias Klein."
  job: "Machine Learning Expert, Data Scientist"
  description: "I spend a lot of time building machine learning solutions"
  btn-text: "Find out more about my skills."
  btn-link: "/projects/"
  profile-image: "assets/img/profile.jpeg"

author_profile:
  title: "Predictive modeling is where I excel at."
  description: ""

social-links:
  twitter: "https://www.kaggle.com/kletobias"
  github: "https://github.com/kletobias"
  youtube: "https://www.youtube.com/@summarizingthingsdatascien3325"
---

## README.md

On this website you can learn more about my skills in the field of machine
learning.

### Overview
The overall idea behind this website is to lower transaction costs by
transparently including my code and descriptions for every step I take in every
article posted on this website.<br> Writing reproducible Python code is an
important aspect for me, along with often using [kaggle
datasets](https://www.kaggle.com/datasets) as input and [kaggle
competitions](https://www.kaggle.com/competitions) for reference scores and real
life machine learning problems.
Entering kaggle competitions provides great feedback on how good one's model
predictions are compared to other competitors.

### Python
Python is the main programming language I use. I am proficient in MySQL for data
import (using `sqlalchemy` to import data for example). You can find a detailed
overview of Python libraries I use in the [Libraries I
Use](TODO:link-to-about-page-libraries-i-use-section) section.

All articles found under [projects]({{ '/projects/' | relative_url }}) belong to one or more
of the following categories.


### data-preprocessing

Articles in this category have a strong focus on transforming *messy tabular
data*, that does not meet the minimum requirements in order to be used as input
to train a machine learning model. Feature engineering is not the focus here,
given that *clean data* is a prerequisite for this step. Below is a selection of
articles I have written on the topic.

[**cleaning a web scraped 47 column pandas dataframe part 3**]({% link _projects/data_prep_3.md %})<br>
[**cleaning a web scraped 47 column pandas dataframe part 4**]({% link _projects/data_prep_4.md %})<br>
[**advanced geospatial feature creation**]({% link _projects/advanced-geospatial-feature-creation.md %})<br>


### deep-learning

[**the math behind \"stepping the weights\"**]({% link _projects/theory_batch_gradient_descent.md %})<br>
[**automation using a test harness for deep learning: part 1**]({% link _projects/1st_tm.md %})<br>
[**automation using a test harness for deep learning: part 2**]({% link _projects/2nd_tm.md %})<br>

### tabular-data

The articles in this category can partially overlap with the ones in the
data-preprocessing category in the way, that data cleaning is mentioned as well.
Apart from the potential overlap, the content of these articles can be about any
step in the machine learning process. E.g., using a deep learning model on
tabular data, feature engineering, cross-validation, or hyperparameter
optimization, to name a few. Below is a selection of articles found under this
category.

[**fill missing values and categorical embeddings**]({% link _projects/tabular_kaggle-1.md %})<br>
[**kaggle submission 1: training RandomForestRegressor and deep learning model**]({% link _projects/tabular_kaggle-6.md %})<br>
[**kaggle submission 2: XGBoost Regressor - RandomSearchCV**]({% link _projects/tabular_kaggle-7.md %})<br><br>
