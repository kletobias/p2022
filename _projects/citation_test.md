---
layout: distill
title: 'hi citatiion beester'
date: 2022-12-29
description: 'testing citations'
img: 'assets/img/838338477938@+-67822330.jpg'
tags: ['deep learning', 'fastai', 'pandas', 'tabular data', 'hypterparameter optimization']
category: ['deep learning']
authors: 'Tobias Klein'
comments: true
---
<br>

A reason why tree based models are one of the best model types when it comes to
understanding the model and the splits a fitted model created lies in the fact,
that there is a good infrastructure for this type of model when it comes to
libraries and functions created for that purpose. That and the fact,
that this type of model has a transparent and fairly easy to understand
structure. The tree based models we use for understanding are ones, where the tree or the
trees in the case of the `RandomForestRegressor` are intentionally kept weak.
This is not necessarily the case for the `DecisionTreeRegressor` model, since
without limiting the number of splits, there is no mechanism to limit the size
of the final tree. (Menze et al., 2009) <d-cite key="menze_comparison_2009"></d-cite> This results in models, that have a low tendency to overfit during
training, which is a characteristic, that is good for the purpose of
understanding the model.

One method, that the `DecisionTreeRegressor` model has, is the
`feature_importances_`. It gives the relative feature importances for all
features.

From the *scikit-learn* website, one gets the following definition for the
`feature_importances_` attribute:

> property feature_importances_<br>
> Return the feature importances.
> The importance of a feature is computed as the (normalized) total
> reduction of the criterion brought by that feature. It is also known as
> the Gini importance.
>
> [*Definition of feature_importances_ attribute on scikit-learn*](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.feature_importances_)

This paper <d-cite key="menze_comparison_2009"></d-cite> can be read for more on
the Gini importance metric.

YES besters <d-cite key="kuhn_applied_2013"></d-cite>

TESSS

<h2>Related Publications</h2>
{% bibliography %}
