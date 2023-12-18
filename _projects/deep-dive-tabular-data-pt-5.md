---
layout: distill
title: 'Addressing the Out-of-Domain Problem in Feature Selection, Part 5'
date: 2023-01-05
description: 'Comprehensive exploration of feature importance in tackling the out-of-domain problem, including identification, significance, and mitigation strategies.'
img: 'assets/img/838338477938@+-791693336.jpg'
tags: ['feature-selection-expertise', 'out-of-domain-solutions', 'model-robustness', 'random-forest', 'tabular-data']
category: ['tabular-data']
authors: 'Tobias Klein'
comments: true
---
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#introduction">Introduction</a></div>
    <div class="no-math"><a href="#understanding-and-addressing-the-out-of-domain-problem-in-machine-learning">Understanding and Addressing the Out-of-Domain Problem in Machine Learning</a></div>
    <div class="no-math"><a href="#identifying-out-of-domain-data-with-randomforestregressor">Identifying Out-of-Domain Data with RandomForestRegressor</a></div>
    <div class="no-math"><a href="#identifying-and-dropping-features-for-improved-model-accuracy">Identifying and Dropping Features for Improved Model Accuracy</a></div>
    <div class="no-math"><a href="#summary">Summary</a></div>
  </nav>
</d-contents>

# Series: Kaggle Competition - Deep Dive Tabular Data
<br>
[**Advanced Missing Value Analysis in Tabular Data, Part 1**]({% link _projects/deep-dive-tabular-data-pt-1.md %})<br>
[**Decision Tree Feature Selection Methodology, Part 2**]({% link _projects/deep-dive-tabular-data-pt-2.md %})<br>
[**RandomForestRegressor Performance Analysis, Part 3**]({% link _projects/deep-dive-tabular-data-pt-3.md %})<br>
[**Statistical Interpretation of Tabular Data, Part 4**]({% link _projects/deep-dive-tabular-data-pt-4.md %})<br>
[**Addressing the Out-of-Domain Problem in Feature Selection, Part 5**]({% link _projects/deep-dive-tabular-data-pt-5.md %})<br>
[**Kaggle Challenge Strategy: RandomForestRegressor and Deep Learning, Part 6**]({% link _projects/deep-dive-tabular-data-pt-6.md %})<br>
[**Hyperparameter Optimization in Deep Learning for Kaggle, Part 7**]({% link _projects/deep-dive-tabular-data-pt-7.md %})<br>
<br>

# Addressing the Out-of-Domain Problem in Feature Selection, Part 5

## Introduction

The "out of domain" problem is a critical challenge in the application of machine learning models to real-world scenarios. This issue arises when a model, trained on a particular dataset, encounters new data with a distribution that significantly deviates from its training environment. Such discrepancies can adversely affect the model's accuracy and reliability, leading to erroneous outcomes that can have far-reaching consequences, particularly in industries where decision-making is heavily reliant on predictive insights.

For instance, in the finance sector, machine learning models play a pivotal role in fraud detection and credit risk assessment. These models need to perform with high precision to prevent substantial financial losses and protect the firm's credibility. Training on non-representative datasets can cause models to misclassify legitimate transactions as fraudulent (false positives) or fail to detect actual fraud (false negatives), both of which can have dire financial and reputational repercussions.

Consequently, it is imperative to tackle the out of domain problem to ensure the efficacy of machine learning deployments in industry settings. Techniques like transfer learning and domain adaptation are employed to mitigate this issue. Transfer learning involves adjusting a pre-trained model to a new dataset, while domain adaptation techniques modify the model or its features to better align with the new domain's characteristics.

Recent research endeavors have introduced a variety of methods aimed at enhancing the generalizability and resilience of machine learning models against the out of domain problem. Significant contributions to this field have been acknowledged<d-footnote>"Unsupervised Domain Adaptation by Backpropagation" by Ganin and Lempitsky</d-footnote><d-footnote>"Domain-Adversarial Training of Neural Networks" by Tzeng et al.</d-footnote><d-footnote>"Deep Domain Confusion: Maximizing for Domain Invariance" by Ghifary et al.</d-footnote>.

## Understanding and Addressing the Out-of-Domain Problem in Machine Learning

In practical terms, the out of domain problem can be illustrated by a simple simulation of data points. The provided code snippet generates a linear sequence of x-values and associates y-values with an added layer of random noise, representing the variability often present in real-world data. The scatter plot of these points helps visualize the potential discrepancies a model might face when applied beyond its training scope.

```python
# Generate a sequence of linear values and corresponding noisy y-values
xlins = torch.linspace(0, 20, steps=45)
ylins = xlins + torch.randn_like(xlins)
# Visualize the data points with a scatter plot
plt.scatter(xlins, ylins)
```

The scatter plot underscores the challenge of modeling under the presence of noise, which can be viewed as a proxy for out-of-domain data. The deviation from the expected linear relationship exemplifies how a model might struggle when predicting on data that differ from the training distribution, emphasizing the necessity for robust machine learning practices that can withstand such variability.






    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_162_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    

However, in order to train an estimator on this data, a second axis is needed.
This can be achieved by using the `.unsqueeze` method or slicing the `xslins`
variable using `None`.

```python
xslins = xlins.unsqueeze(1)
xlins.shape, xslins.shape
```




    (torch.Size([45]), torch.Size([45, 1]))





```python
xlins[:, None].shape
```




    torch.Size([45, 1])


To illustrate the challenges of model generalization and the "out of domain" problem, we train two different regression models: the `RandomForestRegressor` and the `XGBRegressor`. These models are tasked with learning the relationship between the `xslins` and `ylins` data points, which we have synthetically generated to simulate a real-world scenario.

The training is performed using only the first 35 data points of the `xslins` and `ylins` series. This approach is intentional, to demonstrate what happens when models are asked to predict beyond the range they have been trained on – a common situation in practical applications.

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Train a RandomForestRegressor with the first 35 data points
m_linrfr = RandomForestRegressor().fit(xslins[:35], ylins[:35])

# Train an XGBRegressor with the same data points
m_lin = XGBRegressor().fit(xslins[:35], ylins[:35])
```

After training, we use these models to predict values across the entire range of `xslins`, including the final five values that the models were not trained on. We create a scatter plot to visualize these predictions compared to the actual data points. This visualization starkly reveals the out-of-domain problem: for the last five values (the data points beyond what the models have been trained on), both models' predictions are consistently lower than the actual `ylins` values. This discrepancy is a classic example of an extrapolation challenge, where models are prone to inaccurate predictions outside the range of the training data's dependent variable.

```python
# Plotting the model predictions and the actual data points
plt.figure(figsize=(10, 8))
plt.scatter(xslins, m_linrfr.predict(xslins), c="red", alpha=0.4, s=10, label='RandomForestRegressor Predicted Values')
plt.scatter(xslins, m_lin.predict(xslins), c="blue", marker="2", alpha=0.5, label='XGBRegressor Predicted Values')
plt.scatter(xslins, ylins, 20, marker="H", alpha=0.5, c="yellow", label='Set Of All Values')
plt.vlines(x=15.5, ymin=0, ymax=20, linestyle='dashed', alpha=0.7, label='Last Independent Training Value')
plt.title('Visualization Of The Out-Of-Domain Problem')
plt.legend(loc='best')
plt.xlabel('x-Axis')
plt.ylabel('y-Axis')
plt.show()
```

In the resulting plot, we use different markers and colors to distinguish between the predicted values from the RandomForestRegressor (red), XGBRegressor (blue), and the actual set of values (yellow). A vertical dashed line is also drawn at `x=15.5` to indicate the boundary of the last value used in training. This visualization helps in understanding the limitation of machine learning models when they are asked to predict outside the domain of their training data – a cautionary insight for practitioners in the field of machine learning.


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_170_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This visualization helps in understanding the limitation of machine learning models when they are asked to predict outside the domain of their training data – a cautionary insight for practitioners in the field of machine learning.

</div>
    


The reason behind this problem lies in the structure of the
`RandomForestRegressor` and `XGBRegressor` models. These models average the
predictions of several trees for each sample in the training data, and each tree
averages the values of the dependent variable for all samples in a leaf. This
approach can lead to predictions on out-of-domain data that are systematically
too low. To avoid this issue, it is important to ensure that the validation set
does not contain such data.

This out-of-domain problem is a critical issue in machine learning and has
important implications for industry projects where machine learning models are
used to make critical business decisions. Researchers have proposed various
techniques, such as domain adaptation and transfer learning, to address this
problem and improve the generalization and robustness of machine learning
models, as discussed in the literature<d-footnote>Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. In International Conference on Machine Learning (pp. 1180-1189).</d-footnote><d-footnote>Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). Adversarial discriminative domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 7167-7176).</d-footnote><d-footnote>Ghifary, M., Kleijn, W. B., Zhang, M., Balduzzi, D., & Li, W. (2016). Deep domain confusion: Maximizing for domain invariance. arXiv preprint arXiv:1412.3474.</d-footnote>

## Identifying Out-of-Domain Data with RandomForestRegressor

This section describes how to use the `RandomForestRegressor` tool to identify
out-of-domain data. First, the train and validation data are merged, and a new
dependent variable column is added to indicate whether the value of the
dependent variable is part of the train or validation set. Then, the
`RandomForestRegressor` is used to predict whether a row is part of the train or
validation set.

Using the `RandomForestRegressor`, the feature importance of the dataset is
computed, and the top eight most important features are identified. To determine
which features can be dropped without losing accuracy, the RMSE value is
calculated after dropping each of the top eight most important features. It is
observed that dropping the `garagearea` feature does not decrease accuracy, but
dropping `garageyrblt` does. Therefore, only `garagearea` is dropped.

The final RMSE value is then printed, which indicates the accuracy of the model
on the validation set.

This process is important for ensuring the accuracy and reliability of machine
learning models in identifying out-of-domain data, which can lead to incorrect
predictions if not properly identified and handled.

Overall, this section provides a clear explanation of how the
`RandomForestRegressor` tool can be used to identify out-of-domain data and
highlights the importance of this process for the success of machine learning
projects.

### Python Code for Deriving the Findings

The following Python code was used to identify and drop the `garagearea` feature
from the train and validation data, based on the findings discussed in the
previous section. The code demonstrates how the `RandomForestRegressor` tool was
used to identify out-of-domain data, and how feature importance was analyzed to
determine the impact of each feature on model accuracy. By executing this code,
one can replicate the findings presented in the previous section and gain a
deeper understanding of the feature selection process in machine learning
projects.

```python
df_comb = pd.concat([xs_final, valid_xs_final])
valt = np.array([0] * len(xs_final) + [1] * len(valid_xs_final))
m = rf(df_comb, valt)
fi = rf_feat_importance(m, df_comb)
fi.iloc[:5, :]
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
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>lotarea</td>
      <td>0.119988</td>
    </tr>
    <tr>
      <th>1</th>
      <td>grlivarea</td>
      <td>0.108382</td>
    </tr>
    <tr>
      <th>4</th>
      <td>garagearea</td>
      <td>0.098144</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1stflrsf</td>
      <td>0.094516</td>
    </tr>
    <tr>
      <th>8</th>
      <td>totalbsmtsf</td>
      <td>0.092451</td>
    </tr>
  </tbody>
</table>
</div>




```python
m = rf(xs_final, y)
print(f"Original m_rmse value is: {m_rmse(m,valid_xs_final,valid_y)}")
```

    Original m_rmse value is: 0.140712



```python
for c in fi.iloc[:8, 0]:
    m = rf(xs_final.drop(c, axis=1), y)
    print(c, m_rmse(m, valid_xs_final.drop(c, axis=1), valid_y))
```

    lotarea 0.142546
    grlivarea 0.144781
    garagearea 0.137757
    1stflrsf 0.140024
    totalbsmtsf 0.140971
    lotfrontage 0.141214
    bsmtfinsf1 0.143163
    yearremodadd 0.143747


```python
bf = ["garagearea"]
m = rf(xs_final.drop(bf, axis=1), y)
print(m_rmse(m, valid_xs_final.drop(bf, axis=1), valid_y))
```

    0.137757


```python
bf = ["garageyrblt"]
m = rf(xs_final.drop(bf, axis=1), y)
print(m_rmse(m, valid_xs_final.drop(bf, axis=1), valid_y))
```

    0.138856


```python
bf = ["garageyrblt", "garagearea"]
m = rf(xs_final.drop(bf, axis=1), y)
print(m_rmse(m, valid_xs_final.drop(bf, axis=1), valid_y))
```

    0.14009


```python
bf = ["garagearea"]
xs_final_ext = xs_final.drop(bf, axis=1)
valid_xs_final_ext = valid_xs_final.drop(bf, axis=1)
```

## Identifying and Dropping Features for Improved Model Accuracy

The above section discusses the process of identifying and dropping the
`garagearea` feature from the train and validation data. It is confirmed that
using the current value for `random_state`, the `m_rmse` value on the test set
decreases when dropping the `garagearea` feature. The independent part of the
train and validation data is then updated by dropping `garagearea` in both.

To ensure that the column names remain consistent in the future, the column
names for `xs_final_ext` and `valid_xs_final_ext` DataFrames are saved. A
scatter plot of the data in `garagearea` and the distributions of its values
across the train and validation set are then examined. It is observed that the
distribution of `garagearea` has a lower Q1 in the validation set, indicating
that some of the values larger than the Q4 upper bound for the training set are
larger than the ones found in the validation set.

These observations are important because they highlight potential issues with
the data and the need for careful feature selection to ensure accurate and
reliable machine learning models. By dropping the `garagearea` feature, the
accuracy of the model on the validation set is improved.

Overall, this section provides important insights into the process of
identifying and dropping features that may impact the accuracy of machine
learning models. It emphasizes the importance of careful feature selection and
the need for attention to detail when working with complex data. The process
described in this section can help ensure the reliability and accuracy of
machine learning models in industry settings. References in the literature
include<d-footnote>Chollet, F. (2018). Deep Learning with Python. Manning Publications.</d-footnote><d-footnote>Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition. O'Reilly Media, Inc.</d-footnote>.

### A Feature Selection Process for Improved Model Accuracy

The following code section presents a comprehensive implementation of the
feature selection process discussed earlier. The code demonstrates the precise
method used to drop the `garagearea` feature from both the train and validation
data while maintaining consistency in the column names for future use. Moreover,
the code includes a histogram plot and an ECDF plot of the density of values in
`lotarea`, illustrating a case where there is no apparent difference in the
distributions of a variable's values between the train and validation sets. This
visualization is vital in helping professionals identify any issues with the
data and emphasizes the need for meticulous feature selection to ensure precise
and dependable machine learning models. The code section offers a practical
demonstration of the feature selection process and underscores the significance
of attention to detail when working with complex data in machine learning
projects.

```python
for i in ["xs_final_ext", "valid_xs_final_ext"]:
    pd.to_pickle(i, f"{i}.pkl")

finalcolsdict = {
    "xs_final_ext": xs_final_ext.columns.tolist(),
    "valid_xs_final_ext": valid_xs_final_ext.columns.tolist(),
}
```

```python
plt.close("all")
fig = plt.figure(figsize=(6,6))
sns.histplot(data=df_comb, x="lotarea", hue=valt,stat='density',multiple='stack',)
plt.title('Histogram Of "lotarea"\n0 == training data, 1 == validation data')
plt.ylabel('Relative Frequency')
plt.xlabel('Value (Square Feet)')
sns.displot(data=df_comb, x="lotarea", hue=valt,kind='ecdf')
plt.title('ECDF Of "lotarea"\n0 == training data, 1 == validation data')
plt.ylabel('Cumulative Density')
plt.xlabel('Value (Square Feet)')
plt.subplots_adjust(top=.9)
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_186_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        
</div>
    



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <!-- Figure inclusion for the ECDF plot -->
        {% include figure.html path="assets/img/tabular-deep-dive-series/output_187_0.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The ECDF plot reveals the distribution differences for 'lotarea' between the training data (indicated as 0) and validation data (indicated as 1). The cumulative density curves are closely aligned for the majority of the range, indicating a strong distributional similarity. A critical distinction, however, is evident at the higher end of the value spectrum: the training data extends beyond 200,000 square feet, whereas the validation data's 'lotarea' does not exceed 150,000 square feet. This divergence at the upper bounds is emblematic of the out-of-domain problem, highlighting potential challenges in applying a model trained on this data to accurately predict for larger lot areas that are not represented in the validation set.
</div>
    

## Summary
In this article, we explored the process of feature selection in machine
learning projects. We discussed the importance of careful feature selection to
ensure accurate and reliable machine learning models, and demonstrated how to
identify and drop features that may impact model accuracy. We also examined the
out-of-domain problem, which can occur when a model is asked to make predictions
on data outside the range of what it has seen during training. Through practical
examples and visualizations, we emphasized the need for attention to detail when
working with complex data, and provided insights into the feature selection
process that can help ensure the reliability and accuracy of machine learning
models in industry settings. By following these best practices, data
professionals can ensure the success of their machine learning projects and
build models that are both effective and reliable.  

<br>
<br>
<br>
<br>
Entire Series:<br>
<br>
[**Advanced Missing Value Analysis in Tabular Data, Part 1**]({% link _projects/deep-dive-tabular-data-pt-1.md %})<br>
[**Decision Tree Feature Selection Methodology, Part 2**]({% link _projects/deep-dive-tabular-data-pt-2.md %})<br>
[**RandomForestRegressor Performance Analysis, Part 3**]({% link _projects/deep-dive-tabular-data-pt-3.md %})<br>
[**Statistical Interpretation of Tabular Data, Part 4**]({% link _projects/deep-dive-tabular-data-pt-4.md %})<br>
[**Addressing the Out-of-Domain Problem in Feature Selection, Part 5**]({% link _projects/deep-dive-tabular-data-pt-5.md %})<br>
[**Kaggle Challenge Strategy: RandomForestRegressor and Deep Learning, Part 6**]({% link _projects/deep-dive-tabular-data-pt-6.md %})<br>
[**Hyperparameter Optimization in Deep Learning for Kaggle, Part 7**]({% link _projects/deep-dive-tabular-data-pt-7.md %})<br>
<br>
