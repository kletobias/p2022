---
layout: distill
title: 'The Math Behind<br>"Stepping The Weights"'
date: 2023-01-12
description: 'In this article we highlight a key concept in the Stochastic Gradient Descent and explore the basics that this optimization algorithm is derived of.'
img: 'assets/img/838338477938@+-398898935.jpg'
tags: ['deep-learning', 'math', 'ordinary-least-squares', 'partial-derivate', 'stochastic-gradient-descent']
category: ['machine-learning-concepts']
authors: 'Tobias Klein'
comments: true
---
<br>
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div><a href="#a-univariate-linear-regression-function">A Univariate Linear Regression Function</a></div>
    <div class="no-math"><a href="#minimizing-the-l_2-loss">Minimizing The $$L_{2}$$ Loss</a></div>
    <div class="no-math"><a href="#from-loss-function-to-optimization">From Loss Function To Optimization</a></div>
    <div class="no-math"><a href="#univariate-linear-regression">Univariate Linear Regression</a></div>
    <div class="no-math"><a href="#gradient-descent">Gradient Descent</a></div>
    <div class="no-math"><a href="#stochastic-gradient-descent">Stochastic Gradient Descent</a></div>
    <div class="no-math"><a href="#summary">Summary</a></div>
  </nav>
</d-contents>

# The Math Behind<br>"Stepping The Weights"
<br>
**Stochastic Gradient Descent in detail.**

In this article we highlight a key concept in the Stochastic Gradient Descent
and explore the basics that this optimization algorithm is derived of. These
terms and concepts are covered in this article, among others.

1. The components of a univariate linear function (a straight line).
2. What are **weights** in this context?
3. Linear Regression.
4. The empirical loss (using $$\mathit{L}_2$$).
5. Gradient Descent


## A Univariate Linear Regression Function

A basic linear regression function, which is used in machine learning
applications and also in general applications of statistical modeling for
example.<br>
<br>
Looking at it as a mathematical function, it is a single straight line. A linear
function therefore that only has two real-values parameters $$w_{0},\,w_{1}$$.
$$w_{0}$$ is the parameter that specifies the intercept of the function, while
$$w_{1}$$ specifies the slope of the regression line. These parameters are called
**weights**. In addition to these parameters, there is the independent variable,
oftentimes referred to as $$x$$ and $$x_{i}\,i\, \in \mathit{I}$$, if referring to
the elements of $$x$$. The dependent variable is denoted by $$y$$ and the linear
regression $$h_{w}(\,x)\,$$ in this case. The function then looks like this:<br>
<br>

$$h_{w}(\,x)\, =\, w_{1}x\,+\,w_{0}$$

<br>

This function is optimized by the linear regression algorithm, e.g., given $$n$$
training points in the *x,y* plane. **Linear Regression** finds the best fit for
$$h_{w}$$, given these data. The only values that it can change are the ones of
$$w_{0}$$ and $$w_{1}$$, in order to minimize the empirical loss.<br>

## Minimizing The $$L_{2}$$ Loss

If the noise of the dependent variable $$y_{j}$$ is normally distributed, then
using a squared-error loss function will be the most likely type to find the
best values for $$w_{0}$$ and $$w_{1}$$, *that linear regression is capable of
finding.* We assume that the noise of the dependent variable is normally
distributed and use $$L_{2}$$ as loss function. We sum over all the training
values:

$$
\begin{aligned}
\mathit{Loss}(\,h_{w})\,=\,&\sum_{j=1}^{N}\,\mathit{L}_{2}(\,y_{j},\,h_{w}(\,x_{j})\,) \\
\iff\, &\sum_{j=1}^{N}\,(\,y_{j}\, -\, h_{w}(\,x_{j})\,)^2 \\
\iff\, &\sum_{j=1}^{N}\,(\,y_{j}\, -\,(\,w_{1}x{j}\, +\, w_{0})\,)^2\,
\end{aligned}
$$


# From Loss Function To Optimization


```python
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import tensor

%matplotlib inline
```

## Univariate Linear Regression



```python
df = pd.read_csv("houses.csv").drop(columns="Unnamed: 0")
df.sample(n=10)
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
      <th>Price</th>
      <th>Size</th>
      <th>Lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>123500</td>
      <td>1161</td>
      <td>9626</td>
    </tr>
    <tr>
      <th>4</th>
      <td>160000</td>
      <td>2536</td>
      <td>9234</td>
    </tr>
    <tr>
      <th>7</th>
      <td>145000</td>
      <td>1572</td>
      <td>12588</td>
    </tr>
    <tr>
      <th>5</th>
      <td>85000</td>
      <td>2368</td>
      <td>13329</td>
    </tr>
    <tr>
      <th>12</th>
      <td>156000</td>
      <td>2240</td>
      <td>21780</td>
    </tr>
    <tr>
      <th>6</th>
      <td>85000</td>
      <td>1264</td>
      <td>8407</td>
    </tr>
    <tr>
      <th>16</th>
      <td>182000</td>
      <td>1320</td>
      <td>15768</td>
    </tr>
    <tr>
      <th>13</th>
      <td>146500</td>
      <td>1269</td>
      <td>11250</td>
    </tr>
    <tr>
      <th>0</th>
      <td>212000</td>
      <td>4148</td>
      <td>25264</td>
    </tr>
    <tr>
      <th>18</th>
      <td>125000</td>
      <td>1274</td>
      <td>13634</td>
    </tr>
  </tbody>
</table>
</div>



Given price, size of the house and lot, we eliminate lot, so that there is only
one independent variable (size).



```python
df_indep = df["Size"].to_numpy()
df_dep = df["Price"].to_numpy()
```

We create a linear regression model that uses ordinary least squares for
optimization by minimizing $$\mathit{L}_{2}$$, using the implementation in
the `sklearn` machine learning library. `reg.score` gives the global minimum for
the $$\mathit{L}_{2}$$ loss that one achieves using the optimal values
for `reg.coef_` and `reg.intercept_`. $$\mathit{L}_{2}$$ is defined for a single
training example, using `pred` for the model's prediction and `dep` for the value
of the dependent variable that the model tries to predict using the independent
variable and the parameters:

$$\mathit{L}_{2}(\,\mathit{pred},\mathit{dep})\, =\, (\,\mathit{pred}\, -\, \mathit{dep})^2$$

After the model is fitted, the `.score` attribute gives the minimum value of the
loss function using the $$\mathit{L}_2$$ loss.


```python
reg = LinearRegression(fit_intercept=True, n_jobs=-1).fit(
    df_indep.reshape(-1, 1), df_dep
)
reg.score(df_indep.reshape(-1, 1), df_dep)
```




    0.4689809992584135



`reg.coef_` gives the optimal value for the slope parameter,
while `reg.intercept_` gives the optimal value for the intercept parameter.


```python
w1 = reg.coef_
w0 = reg.intercept_

print(
    f"The optimal value for the slope parameter is: {w1},\nwhile {w0} is the optimal value for the intercept."
)
```

    The optimal value for the slope parameter is: [48.19930529],
    while 64553.68328966276 is the optimal value for the intercept.


The function `hw` is a generic univariate linear regression function that makes
the code more reproducible.


```python
def hw(x, w1, w0):
    return w1 * x + w0
```




### Plot Of A Univariate Regression Model


The plot below, shows training examples for independent and dependent variable
and the linear regression line, using the optimal parameters.


```python
fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
x_vals = np.linspace(min(df_indep) + 10, max(df_indep) + 5, 1000)
ax.scatter(df_indep, df_dep, c="r", label="actual x,y pairs")
ax.plot(x_vals, hw(x_vals, w1, w0), c="y", label="regression line")
ax.set_xlabel("Size of the house in square feet")
ax.set_ylabel("Price in USD")
plt.title("Ordinary Least Squares Univariate Linear Regression using $$L_{2}$$")
plt.legend(loc="best")
plt.show()
```


{% include figure.html path="assets/img/output_13_0.png" class="img-fluid rounded z-depth-1" %}



### Plot Of The Loss Surface


Next, we plot the loss surface for $$\mathit{L}_{2}$$. The convex shape of it is
very important, as will be discussed.



```python
def plotls(x, w1, w0):
    slope = np.linspace(w1 - 0.5 * w1, w1 + 0.5 * w1, 20)
    bias = np.linspace(w0 - 0.5 * w0, w0 + 0.5 * w0, 20)
    w1, w0 = np.meshgrid(slope, bias)
    loss = np.power((hw(df_indep, w1, w0) - df_dep), 2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    surface = ax.plot_surface(
        w0,
        w1,
        loss,
        label="$$\mathit{L}_{2}$$ loss surface",
        cmap="viridis",
        edgecolor="none",
    )
    surface._facecolors2d = surface._facecolor3d
    surface._edgecolors2d = surface._edgecolor3d
    ax.set_xlabel("w1 - slope")
    ax.set_ylabel("w0 - bias")
    plt.legend(loc="best")
```

The loss function is convex and thus, always has a global minimum.


```python
plotls(df_indep, w1, w0)
```


{% include figure.html path="assets/img/output_17_1.png" class="img-fluid rounded z-depth-1" %}


### Solving By Hand

The loss function has a gradient of 0, where the minimum is found. The equation,
that has to be solved, fulfills $$w_{opt}\, =\, \mathrm{Min}\,\mathit{Loss}_{w}(
\,h_{w})\,$$. The sum $$\sum_{j=1}^{N}\,(\,y_{j}\, - (\,w_{1}x_{j}\, + w_{0})\,)
^2$$ is minimized, when its partial derivatives with respect to $$w_{0}$$ and $$w_
{1}$$ are zero.
$$
\begin{aligned}&\frac{\partial{h_{w}}}{\partial{w_{0}}}\, \sum_{j=1}^{N}\, (\,y_{j} - (\, w_
{1}x_{j}\, + w_{0})\,)^2 \,= 0 \\
\mathrm{and}\,\, &\frac{\partial{h_
{w}}}{\partial{w_{1}}}\, \sum_{j=1}^{N}\, (\,y_{j} - (\, w_{1}x_{j}\, + w_{0})
\,)^2\, = 0
\end{aligned}
$$

Solving for $$w_{1}$$ and $$w_{0}$$ respectively, gives:

$$
\begin{aligned}&w_{1}\, =\, \frac{\mathit{N}\,(\,\sum_{j=1}^{N}\, x_{j}y_{j})\, - (\,\sum_
{j=1}^{N}\,x_{j})\,(\,\sum_{j=1}^{N}\, y_{j})\,}{\mathit{N}(\,\sum_{j=1}^{N}\,x_
{j}^2)\, - (\,\sum_{j=1}^{N}\, x_{j})^2} \\
&w_{0}\, =\, \frac{(\,\sum_{j=1}^N
y_{i} - w_{1}\,(\,\sum_{j=1}^{N}x_{j})\,)\,}{\mathit{N}}
\end{aligned}
$$

We plug in the values for the independent and dependent variables and solve for
$$w_{1}$$ and $$w_{0}$$ respectively. Two functions are defined, one for each
parameter.


```python
def w1_solve(indep, dep):
    N = len(indep)
    nom = N * np.array(df_indep * df_dep).sum() - (
        np.array(indep).sum() * np.array(dep).sum()
    )
    denom = N * np.array(np.power(indep, 2).sum()) - np.array(
        np.power(np.array(indep).sum(), 2)
    )
    opt = nom / denom
    return opt


def w0_solve(indep, dep):
    N = len(indep)
    nom = dep.sum() - (w1_solve(indep, dep) * (np.array(indep).sum()))
    denom = N
    opt = nom / denom
    return opt
```

Solving for the slope parameter using `w1_solve`.


```python
w1_solve(df_indep, df_dep)
```




    48.19930529470915



In the same way, we use `w0_solve` to solve for $$w_{0}$$.


```python
w0_solve(df_indep, df_dep)
```




    64553.683289662775



The optimal values for $$w_{0}$$ and $$w_{1}$$ that we gained from solving manually
are the same as the ones that `LinearRegression` calculated for the two
parameters. This is confirmed using `np.allclose`.


```python
rig = [w0, w0_solve(df_indep, df_dep), w1, w1_solve(df_indep, df_dep)]


def allclose(rig):
    print(np.allclose(rig[0], rig[1]))
    print(np.allclose(rig[2], rig[3]))


allclose(rig)
```

    True
    True


### Summary: Univariate Linear Model

The univariate linear model always has an optimal solution, where the partial
derivatives are zero. However, this is not always the case, and the following
algorithm for minimizing loss that does not depend on solving for zero values of
the derivatives. It can be applied to any loss function. The **Gradient
Descent** optimization algorithm is that optimizer. Its variation, the
**Stochastic Gradient Descent** is widely used in deep learning to drive the
training process.


## Gradient Descent


The starting point for the Gradient Descent is any point in the weight space.
Here that is a point in the $$(\,w_{0},w_{1})\,$$ plane. One then computes an
estimate of the gradient and moves along the steepest gradient, during each
step. This is repeated, until convergence is reached, which is not guaranteed in
general. The point, on which convergence is reached can be a local minimum loss
or a global one. In detail, while not converged, the Gradient Descent does the
following:


### Gradient Descent: Step


For each weight $$w_{i}$$ in the set of all weights $$\mathbb{w}$$, do for each step:

$$w_{i} = w_{i}\, - \alpha \frac{\partial}{\partial w_{i}}\, \mathit{Loss}(
\,\mathbb{w})\,$$


### Learning Rate


Parameter $$\alpha$$ determines, how large each step size is. It is usually called
**Learning Rate**. It is either a constant or decays over time or changes by
layer.


### Single Training Example


We calculate the partial derivatives, using the chain rule, and a single pair
of independent and dependent variable $$(x,y)$$.

$$\frac{\partial}{\partial w_{i}}\mathit{Loss}(\,\mathbb{w})\, = \,
\frac{\partial}{\partial w_{i}}(\,y\,-h_{w}(\,x)\,)^2 \, =\, 2(\,y\, - h_{w}(
\,x)\,)\, \frac{\partial}{\partial w_{i}} (\,y - (\,w_{1}x + w_{0})\,)\,$$

The partial derivative was not specified for the 'inner function', since it
depends on which of the two $$w$$ parameters is parsimoniously derived.

The partial derivatives to $$w_{0}$$, $$w_{1}$$ are the following:

$$\frac{\partial}{\partial w_{0}}(\,y - (\,w_{1}x + w_{0})\,)\, =
\frac{\partial}{\partial w_{0}} (\,y - w_{1}x - w_{0})\, =\, -1$$

$$\frac{\partial}{\partial w_{1}}(\,y - (\,w_{1}x + w_{0})\,)\, =
\frac{\partial}{\partial w_{1}} (\,y - w_{1}x - w_{0})\, =\, -x$$

Thus, the partial derivative of the loss function for $$w_{0}$$ and $$w_{1}$$ is:

$$
\begin{aligned}&\frac{\partial}{\partial w_{0}}\mathit{Loss}(\,\mathbb{w})\, = -2\,(\,y\, - h_
{w}(\,x)\,) \\
\mathrm{and}\,\, &\frac{\partial}{\partial w_{1}}\mathit{Loss}(
\,\mathbb{w})\, = -2x\,(\,y\, - h_{w}(\,x)\,)\,
\end{aligned}
$$

With these equations calculated, it is possible to plug in the values in the
pseudocode under 'Gradient Descent: Steps'. The -2 is added to the learning
rate.

$$w_{0}\,\leftarrow\,w_{0}\,+\,\alpha\,(\,y\, - h_{w}(\,x)\,)
\,\,\mathrm{and}\,\, w_{1}\,\,\leftarrow\, w_{1}\,+ \alpha\,x\,(\,y\, - h_{w}(
\,x)\,)\,$$


### N Training Examples


In the case of $$N$$ independent and dependent variable pairs, the equations for
updating the weights, are:

$$w_{0}\,\leftarrow\,w_{0}\,+\,\alpha\,\sum_{j=1}^{N}(\,y_{j}\, - h_{w}(\,x_{j})
\,)\,\,\mathrm{and}\,\, w_{1}\,\leftarrow\, w_{1}\,+ \alpha\,\sum_{j=1}^{N}x_
{j}\,(\,y_{j}\, - h_{w}(\,x_{j})\,)\,$$

The aim is to minimize the sum of the individual losses.


### Batch Gradient Descent


The equations for updating the weights are applied after each batch. A batch
consists of a specified number of training examples that are loaded into memory
at once. The **batch gradient descent** for univariate linear regression updates
the weights after each batch. It is computationally expensive, since it sums
over all $$N$$ training examples for every step and there may be many steps, until
the global minimum is reached. If $$N$$ is equal to the number of total elements
in the training set, then the step is called an **epoch**.


## Stochastic Gradient Descent


The **stochastic gradient descent** or **SGD** is a faster variant. It randomly
picks a small subset of training examples at each step, and updates the weights
using the equation under heading 'Single Training Example'. It is common that
the SGD selects a **minibatch** of $$m$$ out of the $$N$$ examples. E.g., Given
$$N=1\mathit{e}4$$ and the minibatch size of $$m=1\mathit{e}2$$, the difference in
order of magnitude between $$N$$ and $$m$$ is 2, which equals a factor of 100 times
less computationally expensive compared to the entire batch for each step.


### Standard Error


The **standard error** of the mean ($$\:\:{\sigma}_{\tilde{X}}\:\:$$) is the
standard deviation of the sample means from the population mean $$\mu$$. As sample
size increases, the distribution of sample means tends to converge closer
together to cluster around the true population mean $$\mu$$.

The standard error can be calculated with the following formula:

$${\sigma}_{\tilde{X}}\, = \frac{\sigma}{\sqrt{N}}$$

Where $${\sigma}_{\tilde{X}}$$ is the standard deviation of the sample mean
(standard error), $${\sigma}$$ the standard deviation of the population and
$${\sqrt{N}}$$ the square root of the sample size.

Therefore, the standard error grows by the root of the sample size. That means,
that given a minibatch of $$m=100$$ training examples and a batch size of
$$N=10000$$, the denominator, using $$N$$ examples for each step is
$$\sqrt{10000}\,=\,100$$, while for the minibatch it is $$\sqrt{100}\,=\,10$$.

That means that the SGD trades being 100 times less computationally expensive
with a 10 times larger standard error for this example.





## Summary

In this article, we started by introducing the function of a univariate linear
regression model and explored its $$L_{2}$$ loss function. We learned that it is
always convex in the case of $$L_{2}$$ being the loss function by plotting the
loss surface. We went on to optimize it, using the ordinary least squares
algorithm, and by hand as well. We did this by forming the partial derivatives
for both parameters and solving for zero for each one. The values obtained for
the parameters were their optima. This was confirmed when we compared the
self-calculated optima with those of the ordinary least squares method.

The batch gradient descent algorithm was explained and in particular how it
updates the weights during each step. We calculated the partial derivatives for
the parameters and used them to show how the weights are updated during each
step. The stochastic gradient descent was introduced and compared to the batch
gradient descent.
<br><br><br>
