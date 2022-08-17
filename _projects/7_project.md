---
layout: page
title: project 7
description: a project with no image
img:
importance: 7
category: work 
---

{% raw %}
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
        label="$\mathit{L}_{2}$ loss surface",
        cmap="viridis",
        edgecolor="none",
    )
    surface._facecolors2d = surface._facecolor3d
    surface._edgecolors2d = surface._edgecolor3d
    ax.set_xlabel("w1 - slope")
    ax.set_ylabel("w0 - bias")
    plt.legend(loc="best")
```
{% endraw %}

### Summary: Univariate Linear Model

The univariate linear model always has an optimal solution, where the partial
derivatives are zero. However, this is not always the case, and the following
algorithm for minimizing loss that does not depend on solving for zero values of
the derivatives. It can be applied to any loss function. The **Gradient Descent**
optimization algorithm is that optimizer. Its variation, the **Stochastic Gradient Descent** is
widely used in deep learning to drive the training process.

## Gradient Descent

The starting point for the Gradient Descent is any point in the weight space.
Here, that is a point in the $$(\,w_{0},w_{1})\,$$ plane. One then computes an
estimate of the gradient and moves along the steepest gradient, during each
step. This is repeated, until convergence is reached. The point, on which
convergence is reached can be a local minimum loss or a global one. In detail,
while not converged, the Gradient Descent does the following:

### Gradient Descent: Steps

For each $$w_{i}$$ in $$\mathbb{w}$$ do for each step:

$$w_{i} = w_{i}\, - \alpha \frac{\partial}{\partial w_{i}}\, \mathit{Loss}(
\,\mathbb{w})\,$$


### Learning Rate

Parameter $$\alpha$$ determines, how large each step size is. It is usually called
**Learning Rate**. It is either a constant or decays over time or changes by
layer.

### Single Training Example

We calculate the partial derivatives, using the chain rule, and a single sample of
independent and dependent variable $$(\,x,y)\,$$.

$$\frac{\partial}{\partial w_{i}}\mathit{Loss}(\,\mathbb{w})\, = \,
\frac{\partial}{\partial w_{i}}(\,y\,-h_{w}(\,x)\,)^2 \, =\, 2(\,y\, - h_{w}(
\,x)\,)\, \frac{\partial}{\partial w_{i}} (\,y - (\,w_{1}x + w_{0})\,)\,$$

The partial derivative was not specified for the 'inner function', since it
depends on which of the two $$w$$ parameters is parsimoniously derived.

The parcial derivatives to $$w_{0}$$, respectively $$w_{1}$$ are the following.

$$\frac{\partial}{\partial w_{0}}(\,y - (\,w_{1}x + w_{0})\,)\, = \frac{\partial}{\partial w_{0}} (\,y - w_{1}x - w_{0})\, =\, -1$$
$$\frac{\partial}{\partial w_{1}}(\,y - (\,w_{1}x + w_{0})\,)\, = \frac{\partial}{\partial w_{1}} (\,y - w_{1}x - w_{0})\, =\, -x$$

Thus, the partial derivative of the loss function for $$w_{0}$$ and $$w_{1}$$ is:

$$\frac{\partial}{\partial w_{0}}\mathit{Loss}(\,\mathbb{w})\, = -2\,(\,y\, - h_{w}(\,x)\,)\,\, \mathrm{and}\,\, \frac{\partial}{\partial w_{1}}\mathit{Loss}(\,\mathbb{w})\, = -2x\,(\,y\, - h_{w}(\,x)\,)\,$$

With these equations calculated, it is possible to plug in the values in the
pseudocode under 'Gradient Descent: Steps'. The -2 is added to the learning
rate.

$$w_{0}\,\leftarrow\,w_{0}\,+\,\alpha\,(\,y\, - h_{w}(\,x)\,)\,\,\mathrm{and}\,\, w_{1}\,\,\leftarrow\, w_{1}\,+ \alpha\,x\,(\,y\, - h_{w}(\,x)\,)\,$$

### N Training Examples

In the case of $$N$$ independent and dependent variable pairs, the equations for updating the weights looks like this:

$$w_{0}\,\leftarrow\,w_{0}\,+\,\alpha\,\sum_{j=1}^{N}(\,y_{j}\, - h_{w}(\,x_{j})\,)\,\,\mathrm{and}\,\, w_{1}\,\leftarrow\, w_{1}\,+ \alpha\,\sum_{j=1}^{N}x_{j}\,(\,y_{j}\, - h_{w}(\,x_{j})\,)\,$$

The aim is to minimize the sum of the individual losses.

### Batch Gradient Descent

The equations for updating the weights are applied after each batch. A batch
consists of a specified number of training examples, that are loaded into memory
at once. The **batch gradient descent** for univariate linear regression updates
the weights after each batch. It is computationally expensive, since it sums
over all $$N$$ training examples for every step and there may be many steps, until
the global minimum is reached. If $$N$$ is equal to the number of total elements
in the training set, then the step is called an **epoch**.

## Stochastic Gradient Descent

The **stochastic gradient descent** or **SGD** is a faster variant. It randomly
picks a small subset of training examples at each step, and updates the weights
using the equation under heading 'Single Training Example'. It is common, that
the SGD selects a **minibatch** of $$m$$ out of the $$N$$ examples. E.g., Given
$$N=1\mathit{E}4$$ and the minibatch size of $$m=1\mathit{E}2$$, the difference in order of magnitude
between $$N$$ and $$m$$ is 2, which equals a factor of 100 times less
computationally expensive compared to the entire batch for each step.

The standard error of the mean ($$\:\:{\sigma}_{\tilde{X}}\:\:$$) is the standard
deviation of the sample means from the population mean $$\mu$$. As sample size
increases, the distribution of sample means tends to converge closer together to
cluster around the true population mean $$\mu$$.

The standard error can be calculated with the following formula:

$${\sigma}_{\tilde{X}}\, = \frac{\sigma}{\sqrt{N}}$$

Where $${\sigma}_{\tilde{X}}$$ is the standard deviation of the sample means (
standard error), $${\sigma}$$ the standard deviation of the population and
$${\sqrt{N}}$$ the square root of the sample size.

Therefore, the standard error grows by the root of the sample size. That means,
that given a minibatch of $$m=100$$ training examples and a batch size of $$N=10000$$,
the denominator, using $$N$$ examples for each step is $$\sqrt{10000}\,=\,100$$, while for the
minibatch it is $$\sqrt{100}\,=\,10$$.

That means, that the SGD can take up to ten times more steps, until it converges,
in order for its standard error
