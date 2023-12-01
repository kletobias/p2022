---
layout: distill
title: 'Enhancing Python Assertions with Custom Messages for Better Debugging'
date: 2023-11-29
description: 'Explore how to enhance your Python testing by adding custom messages to assertions. This post demonstrates a practical example, showcasing the benefits of clearer debugging and improved test readability.'
tags: ['python', 'test', 'pytorch', 'assertion', 'debugging']
category: 'testing'
comments: true
---
<br>

# Enhancing Python Assertions with Custom Messages for Better Debugging

## Introduction
In Python, assertions are a convenient way to ensure that a condition holds true at a specific point in a program. When an assertion fails, it raises an `AssertionError`, halting the execution. However, the default error message may not provide enough context to understand which condition failed, especially in complex tests with multiple assertions. 

This post will demonstrate how to add custom messages to assertions for better debugging, using a test function from the [OpenAI Whisper repository](https://github.com/openai/whisper), which is not my own code and is released under the MIT license.

## The Original Code
Consider the following test function which uses multiple assertions:

```python
import os
import pytest
import torch
import whisper
from whisper.tokenizer import get_tokenizer

@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    assert result["language"] == "en"
    assert result["text"] == "".join([s["text"] for s in result["segments"]])

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    all_tokens = [t for s in result["segments"] for t in s["tokens"]]
    assert tokenizer.decode(all_tokens) == result["text"]
    assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")

    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"]
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8
                assert timing["end"] >= 1.8
                timing_checked = True

    assert timing_checked
```

In this code, if any assertion fails, the default `AssertionError` is raised, but it's unclear which assertion caused the error.

## Adding Custom Messages to Assertions

To improve debugging, we can add a custom message to each assertion. This message is displayed when the assertion fails, providing immediate context about the failure.

Here's the modified code:

```python
import os
import pytest
import torch
import whisper
from whisper.tokenizer import get_tokenizer

@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    assert result["language"] == "en", "Language mismatch in transcription result"
    assert result["text"] == "".join([s["text"] for s in result["segments"]]), "Transcribed text does not match concatenated segment texts"

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription, "'my fellow americans' not found in transcription"
    assert "your country" in transcription, "'your country' not found in transcription"
    assert "do for you" in transcription, "'do for you' not found in transcription"

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    all_tokens = [t for s in result["segments"] for t in s["tokens"]]
    assert tokenizer.decode(all_tokens) == result["text"], "Token decoding does not match original text"
    assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>"), "Decoded timestamps do not start correctly"

    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"], f"Segment timing start ({timing['start']}) is not less than end ({timing['end']})"
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8, f"Timing start for 'Americans' ({timing['start']}) is not <= 1.8"
                assert timing["end"] >= 1.8, f"Timing end for 'Americans' ({timing['end']}) is not >= 1.8"
                timing_checked = True

    assert timing_checked, "Timing check for the word 'Americans' was not performed"
```

## Benefits of Custom Assertion Messages

1. **Immediate Context**: When a test fails, the custom message tells you exactly which condition was not met.
2. **Easier Debugging**: Saves time as developers don't need to inspect each assertion or use a debugger to identify the failed condition.
3. **Better Test Documentation**: Acts as inline documentation, making tests more readable and understandable.

## Conclusion
Adding custom messages to assertions is a simple yet effective way to make your Python tests more robust and maintainable. It aids in quick debugging and ensures that your test suite remains a reliable component of the development process.

