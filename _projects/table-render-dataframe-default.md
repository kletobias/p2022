---
layout: distill
title: 'dataframe table render test'
description: 'The best looking table you have ever seen.'
importance: 7
category: work
---

# We Love Tables. Tables Bring Life.

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
