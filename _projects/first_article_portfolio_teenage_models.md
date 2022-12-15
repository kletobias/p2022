---
layout: distill
title: 'Part 1: Basic Automation For Deep Learning'
date: 2021-01-11
description: 'How to create and use a custom test harness, that automates many steps of the deep learning testing process. It lowers GPU idle time, lets one build more models, test more parameter combinations in less time. The fastai library for deep learning is used throughout this article.'
img: 'assets/img/838338477938@+-67822330.jpg'
tags: ['deep learning', 'fastai', 'automation', 'learning rate', 'loss function', 'stochastic gradient descent', 'binary image classification']
category: ['deep learning']
authors: 'Tobias Klein'
comments: true
---

<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#creating-the-datablock-item">Creating The DataBlock Item</a></div>
    <div class="no-math"><a href="#creating-the-test-harness">Creating The Test Harness</a></div>
    <div class="no-math"><a href="#summary--next-steps">Summary & Next Steps</a></div>
  </nav>
</d-contents>

# Part 1: Basic Automation For Deep Learning

**How to create and use a custom test harness, that automates many steps of the
deep learning testing process. It lowers GPU idle time, lets one build more
models, test more parameter combinations in less time. The
[*fastai*](https://www.fast.ai/) library for deep learning is used throughout this
article.**

We will explore how well a pretrained **ResNet18/ResNet34** image classification
model does on classifying two models, from a photo shoot I did a while ago.
The target labels are 'female' and 'male' and the data was labeled by
filename.

The preprocessing was done using *Adobe Lightroom Classic*, as the images were
exported from Lightroom. A square crop was applied to the images, which resulted
in the center of each image being what was left for the machine learning
process. There were no individual adjustments made during the preprocessing.
Finally, all images were resized to have dimensions of 224×224 pixels. Color
space is 'sRGB'.


## Creating The DataBlock Item


The following code shows how the dataset was loaded into the notebook instance,
and the target labels extracted. The DataBlock object is part of the fastai
library. It can be found in the docs following this link:
[*DataBlock Documentation*](https://docs.fast.ai/data.block.html).


```python
from fastai.test_utils import *
from fastai.vision.all import *
from pathlib import Path
import fastai.vision.models
import fastcore
import itertools
import pandas as pd
import re
```



### Path variable


Creating the `Path` variable, that leads to the directory where the images are
in. The `Path` variable has the class `PosixPath`, a path type found
in `pathlib` Python library, [*pathlib Documentation*](https://pathlib.readthedocs.io/en/pep428/index.html#concrete-paths).



```python
path = Path("/datasets/")
Path.BASE_PATH = path
Path.BASE_PATH.ls()
```




    (#1) [Path('male_female')]




```python
(path / "male_female").ls()[:5]
```




    (#5) [Path('male_female/female_model--2.png'),Path('male_female/female_model-.png'),Path('male_female/female_model-1882.png'),Path('male_female/female_model-1883.png'),Path('male_female/female_model-1884.png')]



### Sample Images

The images are from photo shoots I did for a model agency. The models are
'teenagers'. I created the dataset from two of the shoots, where the male model
and female model both had white t-shirts and blue jeans on. There are a number
of poses included for both models and several similar ones. In particular,
there are images where each one holds an iPhone in front of them and takes
selfies.

There was no active image selection from my part and no image processing, apart
from the raw conversion and the square crop, that was applied the same way to
all images.

The dataset has many 'setup' shots, where I adjusted the aperture, shutter
speed and power of the strobe I used. As a result, there are images, that are
overexposed by two or even three stops, e.g., as can be seen in the center image
of the grid below (**Image[1]**). A large portion of the image suffers from *clipping*. Images
that are overexposed by that much have areas in the image, where all pixel
values are completely white, that means in RGB values $$(255,255,255)$$.

For the deep learning model, that means, that it has less raw input information
for these images, when only looking at the information that can gained directly
from a pixel.

These areas could end up being very valuable for the deep learning model, if
there is a certain pattern to be found in these over exposed areas, that is
useful for classifying the images in 'male' model and 'female' model.

**Image[1]**

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/select_output_54_7.png" title="A 3x3 Grid, That Shows Sample Images Of The Dataset." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Image 1 - 9 random sample images, that show images of both categories. The applied
        crop turns each image into a square image and its length is equal to the
        smaller dimension of the original image. The image in the center of the
        grid shows, that 'setup shots', where the power of the strobe was
        adjusted for example, were included as well. Several images depict the
        model holding a cell phone.
</div>

### Splitting Images By Filename


All images are found inside the 'male_female' directory, a subdirectory of the
'datasets' directory. From the filenames, that were printed out, the structure
used to attach the correct label to each image can be seen. For this binary
image classification problem, the filenames contain the target label.

- Images depicting the male model have filenames, that start with 'male_model'.
- Images depicting the female model have filenames, that start with 'female_model'.

The variable `fname` is assigned to the list of images, that contains all
images. `fname` saved the `Path` variable, and not only the filename +
extension. The next step separates the filename from its path and splits the
images into two groups, the target labels 'female' and 'male'.


```python
fname = (path / "male_female").ls()
print(fname[:5])
```

    [Path('male_female/female_model--2.png'), Path('male_female/female_model-.png'), Path('male_female/female_model-1882.png'), Path('male_female/female_model-1883.png'), Path('male_female/female_model-1884.png')]



```python
dd = {"ff": [], "fm": []}
for nn in fname:
    f = re.search(r"((female_model|male_model)-[^.]*\.png$)", str(nn))
    if f != None:
        if f.group(2) == "female_model":
            dd["ff"].append(f.group(1))
        else:
            dd["fm"].append(f.group(1))

```

### Working With Matched Portions


A `dict` dd is creating, that has to keys: `ff` for found female and `fm` for
found male. Both are assigned an empty list, that appends the filename +
extension for each match.

The regular expression uses `re.search`, a function found in the standard Python
library [*re*](https://docs.python.org/3/library/re.html#). The main reason for
this is that `re.search` returns a `match object`, that makes it easy to work
with the matched strings. In this case `f` was assigned as the match object. If
`f` is of type `None`, no filename was appended to either of the dictionary
keys.

If `type(f) != None` is true, then `f` holds capture groups 1 and 2, that by
design will hold:

- `f.group(1)`: the entire filename + the extension.
- `f.group(2)`: Part of the filename, can be either of the two:
    - 'female_model'
    - 'male_model'

Group 2 was used to assign each filename to one of the two lists, associated
with the dictionary keys

The sample output shows how the items in group 1, labeled 'female' look like.


```python
print(dd["ff"][0:5])
```

    ['female_model--2.png', 'female_model-.png', 'female_model-1882.png', 'female_model-1883.png', 'female_model-1884.png']


Analogously a sample of the items, that get labeled 'male'.


```python
print(dd["fm"][0:5])
```

    ['male_model--10.png', 'male_model--2.png', 'male_model--3.png', 'male_model--4.png', 'male_model--5.png']


The lengths of the lists, that hold the values to each of the two dictionary
keys, are printed out to show the class distribution between the two classes.

The output below shows that the distribution between the classes is close to 50%
for each class with the `male` class having 18 more samples, compared to
the `female` class. The images used in this deep learning article were partly
selected as to test whether around 400 images for each of the two classes would
be enough when used in a transfer learning task.

A model using the **ResNet** architecture seemed as the first choice, as it was
trained using general type images and is believed to generalize exceptionally
well across a wide range of images.


```python
print(f'Number of male matches: {len(dd["fm"])}')
print(f'Number of female matches: {len(dd["ff"])}')
```

    Number of male matches: 434
    Number of female matches: 416





### Creation of the `DataBlock` object.

The `DataBlock` is created using the following items:

- `blocks` uses an
    - `ImageBlock` for the independent variable.
    - `CategoryBlock` for the two target labels: `female` and `male`.
- `get_items` will get the images associated with the filenames.
- `splitter` is set to `RandomSplitter` using `seed=42` for reproducible train,
  validation splits across multiple executions.
- Finally, `get_y` specifies the method used to label all the images in the
  dataset, that the model will predict for each image. `RegexLabeller` does
  exactly the same, as what was done using `re.search`. It is really 'the exact
  same', since it uses `re.search` by default. `"name"` will cause it to use the
  filename as the string to match using the specified pattern.


The portion of the pattern used to label the images is the beginning of the
filename, that can either be `female` or `male`. Because of the precedence rule,
that is applied during a regex search, like the one here, the ordering of the
characters to match is important. If `male` was put before `female`, like
so `r"(male|female)_model-[^.]*\.png$"`, there would be no matches for female at
all, since 'male' as a substring of 'female' would match first. `re.findall` or
the more appropriate function `re.fullmatch` would solve the problem as well.


```python
mf = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=using_attr(RegexLabeller(r"(female|male)_model-[^.]*\.png$"), "name"),
)
```




### Checking The DataBlock For Problems

Using the `summary` method on the newly created `DataBlock` to check for any
errors, that might be present.


*The output of this command was included, since it gives insight into what steps
a DataBlock performs, if no errors are raised.*


```python
mf.summary(source=(path))
```

    Setting-up type transforms pipelines
    Collecting items from /datasets
    Found 850 items
    2 datasets of sizes 680,170
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: partial -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}

    Building one sample
      Pipeline: PILBase.create
        starting from
          /datasets/male_female/male_model-1300.png
        applying PILBase.create gives
          PILImage mode=RGB size=224x224
      Pipeline: partial -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
        starting from
          /datasets/male_female/male_model-1300.png
        applying partial gives
          male
        applying Categorize -- {'vocab': None, 'sort': True, 'add_na': False} gives
          TensorCategory(1)

    Final sample: (PILImage mode=RGB size=224x224, TensorCategory(1))


    Collecting items from /datasets
    Found 850 items
    2 datasets of sizes 680,170
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: partial -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
    Setting up after_item: Pipeline: ToTensor
    Setting up before_batch: Pipeline:
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}

    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: ToTensor
        starting from
          (PILImage mode=RGB size=224x224, TensorCategory(1))
        applying ToTensor gives
          (TensorImage of size 3x224x224, TensorCategory(1))

    Adding the next 3 samples

    No before_batch transform to apply

    Collating items in a batch

    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
        starting from
          (TensorImage of size 4x3x224x224, TensorCategory([1, 1, 0, 0], device='cuda:0'))
        applying IntToFloatTensor -- {'div': 255.0, 'div_mask': 1} gives
          (TensorImage of size 4x3x224x224, TensorCategory([1, 1, 0, 0], device='cuda:0'))


No errors were detected and so the `dataloaders` object can be created from the
`DataBlock` using the `path` to the dataset, and the chosen size of the
validation dataset, given as a fraction of the total dataset. The dataloaders
Documentation can be found following this link to its Documentation page:
[*dataloaders Documentation*](https://docs.fast.ai/data.load.html).

Since the image count in this dataset is on the lower side, compared to some
well known datasets, a high value for `valid_pct` of 40% of the total dataset is
chosen to test right away, if there are any signs for overfitting found.
Changing `valid_pct` did not show to produce differing prediction results on the
validation set. The model's predictions remained the same.


```python
dls = mf.dataloaders(path, valid_pct=0.4)
```

## Creating The Test Harness

### List Of Parameter Combinations To Test

For this article, the following combinations of `model`, `valid_pct` and
`fine_tune` were assessed. That meant, given 2 possible settings for each
variable, that a total of 8 configurations had to be tested. The metric used, is
the `error_rate` for all setups.

**Table[1]**

| `model`  | `valid_pct` | `fine_tune` |
|----------|-------------|-------------|
| resnet34 | 0.2         | 1           |
| resnet34 | 0.2         | 2           |
| resnet34 | 0.4         | 1           |
| resnet34 | 0.4         | 2           |
| resnet18 | 0.2         | 1           |
| resnet18 | 0.2         | 2           |
| resnet18 | 0.4         | 1           |
| resnet18 | 0.4         | 2           |

### Error Rate: What It Measures

The error rate measures the fraction of the predictions made by the model, that
are not correct. It takes the absolute number of incorrect predictions and
divides them by the total number of predictions. If the result is supposed to be
in percent, then the fracture has to be multiplied by 100. In the fastai
library, the output of the `error_rate` ([*error_rate Documentation*](https://docs.fast.ai/metrics.html#error_rate)) is the raw fraction, not the percentage. Its numbers are the inverse of the `accuracy` metric. The formula of the
error rate is:

$$
\mathrm{Error\, Rate} := \frac{\mathrm{wrong\,predictions}}{\mathrm{total\,predictions}} 100 \iff 1\, -\, \mathrm{Accuracy}
$$


### Detailed Construction Of The Test Harness


Given, that 8 different configurations in total have to be tested and logged
(more on that later), it certainly is feasible to log everything manually.
However, specifying each configuration manually and not logging the results
using a single DataFrame, that can append any number of structured empirical
experiments, is neither scalable, nor reproducible. To create and log structured
empirical experiments, a test harness is simply the best option.


**The metrics to track are:**

`model` - The specific model used in the configuration. This is
either **ResNet34** or **ResNet18** in this case.

`fine_tune` - The number of epochs used for transfer learning of the pretrained
model. This excludes the initial training epoch, where only the final layers are
trained, and all other layers are frozen. In this case, that is either 1 or 2
epoch(s).

`valid_pct` - Sets the percentage of the dataset, that is not used for training,
but only for validation. Given the relatively low total image count in this
dataset, it is of interest to see what the results are using the default
valid_pct of 0.2 are compared to a valid_pct of 0.4, double the default
valid_pct. The samples in the validation set are unknown to the model and are
only used once to gauge how well the model can predict the target label on
unseen data.

**The values collected for each configuration are:**

`train_loss` and `valid_loss` - The loss function is 'FlattenedLoss of
CrossEntropyLoss' for all configurations.

`error_rate` - The metric chosen to assess the model on the validation set,
using a *metric designed for human consumption*.

I chose to create a test harness, that automatically creates each configuration
specified in the `harness_input` dictionary. It also logs the most important
training values for each configuration. That is, the loss function and metric
and saves it in a dictionary `harness_output`, which at the end is automatically
converted to a `pandas.DataFrame` (`df`). The DataFrame needs no further
processing, since it already fulfills the requirements of *tidy data*.

The way the logging was accomplished, using very limited knowledge of how
`CallBacks` work in the fastai library
([*CallBacks Documentation*](https://docs.fast.ai/callback.core.html)) and what
functions, methods overlap with the ones found in the `PyTorch` library and what
`PyTorch` code can be used inside `fastai`.

There are two callbacks used for logging `train_loss`, `valid_loss` and
`error_rate` for each setup and epoch. Using the `cbs` parameter, a list
containing the two Callbacks `TrainEvalCallback` and `Recorder` was added to the
instantiation call of the *vision_learner* object
([*vision_learner Documentation*](https://docs.fast.ai/vision.learner.html#vision_learner)).
In detail, the call to vision_learner looks as follows:

```python
learn=vision_learner(dls,setup[0],metrics=error_rate,cbs=[TrainEvalCallback,Recorder])
```

The dictionary `harness_input` has keys for all input parameters, that will be
tested and logged. Not all of its keys are used in this article. It is important
not to wrap the values for `harness['model']` in quotes, e.g.,
`models.resnet34`. The value is recognized as being a PyTorch function and will
be accepted, when passed to the `vision_learner` creation.
`input_harness['valid_pct']` has a list of values for the `valid_pct` parameter,
that should all be tested. `harness_input['fine_tune']` specifies the number of
epochs, that should be used for training, after the initial fit one cycle epoch.




```python
harness_input={'learner': [],'model': [], 'fine_tune': [],'lr': [], 'valid_pct': []}
```

Below are the key, value pairs for the parameters that are tested in the
following.


```python
harness_input['model'] = [models.resnet34, models.resnet18]
harness_input['valid_pct']= [0.2,0.4]
harness_input['fine_tune']= [1,2]
```

### Calculation Of Test Setups

To rigorously test each parameter, only one parameter should be changed at a
time.

That means, that two out of the three parameters will remain the same from one
setup to the next one. This description does not take into account ordering, but
it gives the idea behind this empirical testing scheme. It guarantees that all
three element long parameter combination tuples get tested.

The math behind calculating all combinations for any given number of parameters
$$m$$, that have a number of unique parameter values given by $$p_{i}$$, for the
$$i_{th}$$ out of the total $$m$$ parameters. Which can be interpreted as a
vector of length $$m$$, with each element the total number of parameter values
for one of the parameters: $$\langle p_{1},..,p_{m}\rangle$$

Then, the total number of combinations, that have to be tested is given by:

$$\mathrm{number\, of\, combinations}:=\prod_{i=1}^m p_{i}$$

In this case, the calculation is the following, using `model`, `valid_pct` and
`fine_tune`, and their values from Table[1]:

$$2\times 2\times 2\, = \, 8$$

`itertools.product` does exactly that for us and will return a list of all the
combinations of the input lists. See the sample output below.


```python
setups = list(
    itertools.product(
        harness_input["model"], harness_input["valid_pct"], harness_input["fine_tune"]
    )
)
print(len(setups))
setups[0]
```

    8

    (<function torchvision.models.resnet.resnet34(*, weights: Optional[torchvision.models.resnet.ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> torchvision.models.resnet.ResNet>,
     0.2,
     1)



One can use list indexing to access the individual parameter values.

```python
setups[0][2]
```


    1



The output for each setup is logged in a dictionary, that has *keys* for all
parameters that are logged.

For each of the 8 setups, one row per epoch is used. With the setups, that use 2
epochs using 2 rows. A column, that shows, which setup and epoch the other
columns belong to, is added in the form of column `unique_setup`.

The regular expressions used to match the exact name of the model was necessary,
due to the `model` string not only showing the model as a string, but many other
characters as well. E.g., see the `setups[0]` sample output above. The regular
expressions used to capture the valid_pct and fine_tune strings could have been
replaced with the actual values for these two parameters, using `setup[1]` and
`setup[2]` as its values. I prefer the added flexibility that might come in
handy, when using this logging framework on other problems. If it shows, that
there is no benefit to using regular expressions for this, then the direct
assignment will be used instead.

The `cbs=[TrainEvalCallback,Recorder]` in the `vision_learner` assignment was
necessary, since `recorder` kept throwing an error after the first loop. E.g.,

> TypeError: Exception occurred in `Recorder` when calling event `after_batch`:
'float' object is not callable

Adding `TrainEvalCallback` to the list of cbs solved the issue. The `plot_loss`
function, while it could be useful in theory, does not add much insight into
which setup performed well or bad. I found no way to add custom labels to the
line plot and so there is no telling, which lines represent a given setup in
image[2].


```python
harness_output = {
    "unique_setup": [],
    "model": [],
    "fine_tune": [],
    "valid_pct": [],
    "train_loss": [],
    "valid_loss": [],
    "error_rate": [],
    "lr": [],
}
fig,ax=plt.subplots(1,1,figsize=(10,10))
w = 0
for setup in setups:
    for tu in range(0,setup[2]):
        harness_output['unique_setup'].append(str(w) + str('-') +str(tu))
        m=re.search(r'(resnet\d{2})',str(setup[0]),flags=re.IGNORECASE)
        harness_output['model'].append(m.group(0))
        m=re.search(r'(0\.[42])',str(setup[1]))
        harness_output['valid_pct'].append(m.group(0))
        m=re.search(r'([12])',str(setup[2]))
        harness_output['fine_tune'].append(m.group(0))
        dls=mf.dataloaders(path,valid_pct=setup[1])
        learn=vision_learner(
                dls,setup[0],metrics=error_rate,cbs=[TrainEvalCallback,Recorder]
        )
        harness_output['lr'].append(learn.lr)
        learn.fine_tune(setup[2])
        learn.recorder.plot_loss()
        vals=L(learn.recorder.values)
        harness_output['train_loss'].append(vals[0][0])
        harness_output['valid_loss'].append(vals[0][1])
        harness_output['error_rate'].append(vals[0][2])
    w += 1
```


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/output_31_98.png" title="Plot Of Loss Functions On Train And Validation Sets" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Image[2]: Changes in the loss function. Plot shows, that additional parameters
        need to be specified, in order to make this plot of any use.
</div>

The final DataFrame is created from the harness_output dictionary. Its columns
are the keys of the dictionary, with each row holding all the data for one
epoch.

```python
df = pd.DataFrame.from_dict(harness_output)
df
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
      <th>unique_setup</th>
      <th>model</th>
      <th>fine_tune</th>
      <th>valid_pct</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>lr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-0</td>
      <td>resnet34</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.048804</td>
      <td>0.002172</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1-0</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.066204</td>
      <td>0.000746</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1-1</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.053110</td>
      <td>0.002471</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2-0</td>
      <td>resnet34</td>
      <td>1</td>
      <td>0.4</td>
      <td>0.060802</td>
      <td>0.001054</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3-0</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.071246</td>
      <td>0.001824</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3-1</td>
      <td>resnet34</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.062736</td>
      <td>0.001016</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4-0</td>
      <td>resnet18</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.056334</td>
      <td>0.005576</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5-0</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.047683</td>
      <td>0.003617</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5-1</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.2</td>
      <td>0.068953</td>
      <td>0.023827</td>
      <td>0.011765</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6-0</td>
      <td>resnet18</td>
      <td>1</td>
      <td>0.4</td>
      <td>0.055616</td>
      <td>0.012279</td>
      <td>0.005882</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7-0</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.060576</td>
      <td>0.003479</td>
      <td>0.000000</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7-1</td>
      <td>resnet18</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.078422</td>
      <td>0.010864</td>
      <td>0.005882</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>

## Summary & Next Steps

The fastai library holds some incredibly powerful tools, that can be easily used
to create production ready models. It was only because of the pretrained models,
that the fastai library has to offer, that a model could be created, that in
almost all the tested setups has an impeccable error_rate of 0 on the validation
dataset.

In this article, we went from raw images, to creating a
`Path` object, that points at the image files. Using the `RegexLabeller`, we
labeled the images by extracting the labels from the filenames. With that, we
first created a DataBlock and from that a dataloaders object.

This made it possible for us to instantiate a vision_learner object, that used
one of two **ResNet** architecture variants. The list of setups was created, setups
were built and logged. At this point we have a *tidy* DataFrame with all the
logged data.

In *Part 2*, we analyze the logged data in the DataFrame and build more
combinations (*Batch No. 2*) to answer our questions, that we have after
analyzing the DataFrame with the results of *Batch No. 1*.
