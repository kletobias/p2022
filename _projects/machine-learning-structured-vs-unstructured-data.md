---
layout: distill
title: 'Optimal Machine Learning Models for Structured vs Unstructured Data: A Comparative Analysis'
date: 2023-12-13
description: 'Structured data is searchable, suitable for traditional ML. Unstructured data suits deep learning models.'
img: 'assets/img/838338477938@+-398898935.jpg'
tags: ['machine-learning', 'structured-data', 'unstructured-data', 'video-presentation', 'comparison']
category: ['Machine Learning Process']
authors: 'Tobias Klein'
comments: true
---
<br>

<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#video">Video</a></div>
    <div class="no-math"><a href="#structured-data">Structured data</a></div>
    <div class="no-math"><a href="#unstructured-data">Unstructured data</a></div>
    <div class="no-math"><a href="#differences">Differences</a></div>
    <div class="no-math"><a href="#machine-learning">Machine Learning</a></div>
  </nav>
</d-contents>

# Optimal Machine Learning Models for Structured vs Unstructured Data: A Comparative Analysis

## Video

This is the video version of this article, as found on my YouTube channel:

<iframe width="660" height="415" src="https://www.youtube.com/embed/0nWHOFsYxY4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
<br>

## Structured data

Structured data is data that has a high degree of organization and is easily searchable by simple, straightforward search engine algorithms or other search operations. It refers to data that is stored in databases, in a fixed field within a record or file. Examples include data in relational databases, such as spreadsheets, or can be anything from a name, a digital reading, a date, or a fact <d-footnote>Chen, M., Mao, S., & Liu, Y. (2014). Big Data: A Survey. Mobile Networks and Applications, 19(2), 171–209. https://doi.org/10.1007/s11036-013-0489-0</d-footnote>.

## Unstructured data

Unstructured data, on the other hand, is data that is not organized in a pre-defined manner or does not have a pre-defined data model, thus it is not as straightforward to search and analyze. It is typically text-heavy but may also contain data like dates, numbers, and facts. Examples of unstructured data include text files like Word documents, email, social media posts, video, audio files, websites, and more<d-footnote>Sikos, L. F. (2017). Mastering Structured Data on the Semantic Web: From HTML5 Microdata to Linked Open Data. Apress. https://doi.org/10.1007/978-1-4842-1867-1</d-footnote>.

## Differences

The main differences between structured and unstructured data are:

1. **Format**: Structured data is typically organized in a tabular format with clear definitions, whereas unstructured data does not follow any specific format.

2. **Searchability**: Due to its organized format, structured data can easily be searched, while unstructured data, due to its lack of a specific format, is difficult to search.

3. **Scalability**: Unstructured data is often more scalable as it encompasses a broader range of data types, compared to structured data which is limited by its defined structure<d-footnote>Jagadish, H. V., Gehrke, J., Labrinidis, A., Papakonstantinou, Y., Patel, J. M., Ramakrishnan, R., & Shahabi, C. (2014). Big data and its technical challenges. Communications of the ACM, 57(7), 86–94. https://doi.org/10.1145/2611567</d-footnote>.

## Machine Learning

When it comes to machine learning models, certain models tend to be more effective with certain types of data.

### Models for structured data
For structured data, traditional machine learning algorithms like Linear Regression, Decision Trees, Support Vector Machines, or ensemble methods like Random Forests and Gradient Boosting Machines, often perform well. These models can handle the tabular nature of structured data and make use of the relationships between different features<d-footnote>Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Series in Statistics. https://doi.org/10.1007/978-0-387-84858-7</d-footnote>.

### Models for unstructured data
On the other hand, unstructured data is often best processed with deep learning models. Convolutional Neural Networks (CNNs) are commonly used for image and video data. Recurrent Neural Networks (RNNs), particularly those with Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) cells, are typically used for sequential data like text or time series<d-footnote>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org</d-footnote>. These models can process the complex, non-tabular nature of unstructured data and extract hierarchical features.
<br><br><br><br>
