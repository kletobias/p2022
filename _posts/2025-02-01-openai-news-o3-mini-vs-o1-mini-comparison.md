---
layout: distill
title: 'OpenAI News: o3-mini vs o1-mini comparison'
date: 2025-02-01
description: 'Comparing the o3-mini and the o1-mini in terms of response latency, completion tokens, reasoning tokens'
tags: ['openai', 'o3-mini', 'reasoning-models', 'o1-mini', 'reasoning_effort']
category: 'openai'
comments: true
---

<!--_posts/2025-02-01-openai-news-o3-mini-vs-01-mini-comparison.md-->
<br>

# o3-mini VS o1-mini Comparison

The new 3o-mini reasoning model tested!

## Links to Docs

o3-mini: [https://platform.openai.com/docs/models#o3-mini](https://platform.openai.com/docs/models#o3-mini)  
o1-mini: [https://platform.openai.com/docs/models#o1](https://platform.openai.com/docs/models#o1)  
Reasoning models: [https://platform.openai.com/docs/guides/reasoning](https://platform.openai.com/docs/guides/reasoning)  
## LinkedIn Post

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7291446954620780544" height="802" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

## Summary

I’ve been testing the new reasoning_effort parameter in Chat Completions across o3-mini and o1-mini. According to OpenAI, o3-mini is on par with o1-mini for cost per token and latency. My results suggest:

- o1-mini has no adjustable reasoning setting but, in practice, performed similarly to o3-mini set to medium reasoning.
- Low reasoning on o3-mini used significantly fewer reasoning tokens than o1-mini.
- High reasoning on o3-mini used far more reasoning tokens (and took longer).
- Both models share the same knowledge cutoff of October 2023, but we haven’t fully explored all comparisons yet.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/o1-o3-compare.png" title="Comparison between o3-mini-2025-01-31, and the o1-mini-2024-09-12" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 1: Comparison between o3-mini-2025-01-31, and the o1-mini-2024-09-12 in terms of latency, Completion Tokens, and Reasoning Tokens.
</div>
[**Full size image of the plot**]({% link assets/img/o1-o3-compare.png %})

## Experiment Details

### Implementation details:

#### Main Script

Main script – Demonstrates usage of the reasoning_effort parameter and logs completions (including latency and token usage).

```python
# comments_and_posts/run_o3-and-o1-mini-latency.py
import json
import time
from openai import OpenAI
from typing import Literal
from datetime import datetime
import os
from typing import Optional

client = OpenAI()

def get_current_date() -> str:
    """Return current date as a string."""
    return datetime.now().strftime("%Y-%m-%d")

def save_completion(timestamp: str,
                    response: object,
                    filename: Optional[str]) -> None:
    """Save the completion to a JSON file."""
    response_dict = response.to_dict()
    if filename:
        filename_unique = filename
    else:
        filename_unique = f"completion_log_{timestamp}.json"
    with open(filename_unique, "a") as file:
        json.dump(response_dict, file)
        file.write("\n")

def extract_and_save_usage(timestamp: str,
                           reasoning_effort: str,
                           response: object,
                           latency: float) -> None:
    """Extract usage data and save to a JSON file."""
    response_dict = response.to_dict()
    usage_data = {
        "model": response_dict.get("model", None),
        "reasoning_effort": reasoning_effort,
        "latency": latency,
        "prompt_tokens": response_dict["usage"]["prompt_tokens"],
        "completion_tokens": response_dict["usage"]["completion_tokens"],
        "total_tokens": response_dict["usage"]["total_tokens"],
        "prompt_tokens_details": response_dict["usage"].get("prompt_tokens_details", {}),
        "completion_tokens_details": response_dict["usage"].get("completion_tokens_details", {}),
    }
    usage_file = f"usage_log_{timestamp}.json"
    mode = "a" if os.path.exists(usage_file) else "w"
    with open(usage_file, mode) as file:
        json.dump(usage_data, file)
        file.write("\n")

def api_call_with_reasoning(timestamp: str,
                            client,
                            model: str,
                            reasoning_effort: Literal["low", "medium", "high"],
                            prompt: str) -> None:
    """Call the API with the specified reasoning effort."""
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start_time
    save_completion(timestamp, response, filename="")
    extract_and_save_usage(timestamp, reasoning_effort, response, latency)
    print(f"Response for reasoning effort '{reasoning_effort}':")
    print(response.choices[0].message.content)

def api_call_without_reasoning(timestamp: str,
                               client,
                               model: str,
                               prompt: str) -> None:
    """Call the API without specifying reasoning effort."""
    reasoning_effort = "o1-mini default"
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    latency = time.perf_counter() - start_time
    save_completion(timestamp, response, filename=f"completion_no_reasoning_log_{timestamp}.json")
    extract_and_save_usage(timestamp, reasoning_effort, response, latency)
    print("Response without reasoning_effort:")
    print(response.choices[0].message.content)

def run_api_for_all_reasoning_efforts(timestamp: str,
                                      client,
                                      model: str,
                                      prompt: str) -> None:
    """Run the API for low, medium, and high reasoning."""
    reasoning_efforts = ["low", "medium", "high"]
    for effort in reasoning_efforts:
        api_call_with_reasoning(timestamp, client, model, effort, prompt)

def run_api_without_reasoning(timestamp: str,
                              client,
                              model: str,
                              prompt: str) -> None:
    """Run the API without reasoning effort."""
    api_call_without_reasoning(timestamp, client, model, prompt)

prompt = """
Write a bash script that takes a matrix represented as a string with 
format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
"""
model_with_reasoning = "o3-mini"
model_without_reasoning = "o1-mini"
timestamp = get_current_date()

run_api_for_all_reasoning_efforts(timestamp, client, model_with_reasoning, prompt)
run_api_without_reasoning(timestamp, client, model_without_reasoning, prompt)
```

#### Data Analysis Script

Data analysis script – Reads JSON logs, loads them into a DataFrame, then visualizes latency vs. reasoning tokens with pandas + seaborn.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import sys

data = [
  {
    "model": "o3-mini-2025-01-31",
    "reasoning_effort": "low",
    "latency": 7.114629958989099,
    "prompt_tokens": 44,
    "completion_tokens": 990,
    "total_tokens": 1034,
    "completion_tokens_details": {"accepted_prediction_tokens": 0,"audio_tokens": 0,"reasoning_tokens": 192,"rejected_prediction_tokens": 0}
  },
  {
    "model": "o3-mini-2025-01-31",
    "reasoning_effort": "medium",
    "latency": 10.857935665990226,
    "prompt_tokens": 44,
    "completion_tokens": 2331,
    "total_tokens": 2375,
    "completion_tokens_details": {"accepted_prediction_tokens": 0,"audio_tokens": 0,"reasoning_tokens": 1472,"rejected_prediction_tokens": 0}
  },
  {
    "model": "o3-mini-2025-01-31",
    "reasoning_effort": "high",
    "latency": 26.10190358303953,
    "prompt_tokens": 44,
    "completion_tokens": 3700,
    "total_tokens": 3744,
    "completion_tokens_details": {"accepted_prediction_tokens": 0,"audio_tokens": 0,"reasoning_tokens": 2624,"rejected_prediction_tokens": 0}
  },
  {
    "model": "o1-mini-2024-09-12",
    "reasoning_effort": "o1-mini default",
    "latency": 14.15126500010956,
    "prompt_tokens": 46,
    "completion_tokens": 2386,
    "total_tokens": 2432,
    "completion_tokens_details": {"accepted_prediction_tokens": 0,"audio_tokens": 0,"reasoning_tokens": 1408,"rejected_prediction_tokens": 0}
  }
]
df = pd.DataFrame(data)
print(df.columns.tolist())

# sys.exit()
df["reasoning_tokens"] = df["completion_tokens_details"].apply(lambda x: x["reasoning_tokens"])

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(x="model", y="latency", hue="reasoning_effort", data=df, ax=axs[0])
axs[0].set_title("Latency Comparison")
sns.barplot(x="model", y="completion_tokens", hue="reasoning_effort", data=df, ax=axs[1])
axs[1].set_title("Completion Tokens")
sns.barplot(x="model", y="reasoning_tokens", hue="reasoning_effort", data=df, ax=axs[2])
axs[2].set_title("Reasoning Tokens")

for ax in axs:
    ax.set_label("")
plt.tight_layout()
plt.show()
```

Thank you very much for reading this article. Please feel free to link to this article or write a comment in the comments section below.

---

**© Tobias Klein 2025 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
