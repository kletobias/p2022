---
layout: distill
title: "Shocking Hidden Reasoning: OpenAI Flagship Reasoning Model o3-pro"
date: 2025-07-17
description: "Instead of generating the requested dataclasses, the model plans to return a soup of placeholders and dubious parameters - before refusing to reply at all."
tags: ["llm", "openai", "o3-pro", "reasoning", "garbage-output", "sabotage"]
category: "LLM Reasoning"
comments: true
---

<!-- _posts/shocking_hidden_reasoning_openai_flagship_reasoning_model_o3_pro.md -->
<br>

# Shocking Hidden Reasoning: OpenAI Flagship Reasoning Model o3-pro

**🤖 When an LLM quietly boycotts your request**

## Prompt:

> Create the complete @dataclass for the following LangChain loaders class names:
>
> ```python
> langchain_agentql.document_loaders.AgentQLLoader
> langchain_airbyte.AirbyteLoader
> langchain_apify.ApifyDatasetLoader
> langchain_astradb.AstraDBLoader
> langchain_box.blob_loaders.BoxBlobLoader
> langchain_box.document_loaders.BoxLoader
> ```

## o3-pro Reasoning

**Model’s hidden “reasoning”:**

> Mapping out dataclasses…
> I’m evaluating different methods to create dataclasses, considering generating skeletons with placeholders, dubious parameters, or pseudocode annotations.

## Reply

3 minutes later, the model outputs the following Python script

```python
#!/usr/bin/env python
# generate_loader_dataclasses.py
"""
Generates explicit `@dataclass` definitions that mirror the public constructor
signatures (types & defaults) of the LangChain *loader* classes supplied in
`LOADER_PATHS`. Run once in an environment where the target packages (and any
optional dependencies you care about) are import‑able:

    $ python generate_loader_dataclasses.py

The script will create `loader_dataclasses.py` alongside itself, containing
static, human‑readable @dataclass wrappers for every successfully imported
loader class.

Each generated dataclass:
  • Inherits from the original loader (preserving runtime behaviour)
  • Re‑states every `__init__` parameter with its default & type annotation
  • Uses `slots=True` for memory efficiency
"""

# ...
# --------------------------------------------------------------------------- #
# 3. Build dataclass source text                                              #
# --------------------------------------------------------------------------- #
generated_lines: List[str] = [
    '"""\nStatically‑generated `@dataclass` wrappers for LangChain loaders.\n'
    "This file is **auto‑generated** by `generate_loader_dataclasses.py`.\n"
    'Do **NOT** edit manually.\n"""\n',
    "from __future__ import annotations\n",
    "from dataclasses import dataclass, field\n",
    "from typing import Any\n\n",
]

for path in LOADER_PATHS:
    try:
        orig_cls = _import_cls(path)
        base_alias = f"{orig_cls.__name__}Base"
        # alias original class so we can inherit
        generated_lines.append(f"from {path.rsplit('.',1)[0]} import {orig_cls.__name__} as {base_alias}\n")
        fields_spec = _iter_fields(orig_cls)
        generated_lines.append(_render_dataclass(orig_cls.__name__, fields_spec))
    except Exception as exc:
        generated_lines.append(
            f"# Skipped {path!r} due to import error: {exc!r}\n"
        )

# --------------------------------------------------------------------------- #
# 4. Write to disk                                                            #
# --------------------------------------------------------------------------- #
outfile = pathlib.Path(__file__).with_name("loader_dataclasses.py")
outfile.write_text("\n".join(generated_lines))
print(f"[✓] Wrote {outfile.relative_to(outfile.parent)} with {len(LOADER_PATHS)} definitions.")
```

## Analysis of the Reply

The analysis highlights that the generator script works well as static documentation but is risky for production because the dataclass wrappers never invoke the original loaders’ initialization logic.

| Area                 | Impact                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| Runtime behaviour    | The auto-generated `__init__` bypasses the original loader’s side-effects, leaving objects half-configured.  |
| Super call           | Lacking a `__post_init__` that calls `super().__init__`, required setup steps never execute.                 |
| Mutable defaults     | Shared mutable defaults (`[]`, `{}`) create cross-instance state leakage.                                    |
| Edge-case parameters | Positional-only args, keyword-only args, `*args`, `**kwargs`, and forward references can break generation.   |
| Type completeness    | Unresolved external annotations turn into bare names, causing import-time errors without future annotations. |
| Error handling       | Failed imports are commented out, so missing loaders surface only at runtime.                                |
| Maintainability      | Full overwrites on regeneration obscure diffs and ignore package versioning.                                 |

Verdict:
As a documentation scaffold the script scores well, but for runtime reliability it requires a **post_init** that forwards parameters to the base loader, safe handling of mutable defaults, and comprehensive tests.

## Alignment with the task

| Requirement (prompt)                                        | What the reply delivers                                                                                              | Alignment    |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Return full `@dataclass` definitions**                    | Supplies a _generator script_ that can emit dataclasses, but no actual dataclasses are included in the reply.        | ❌           |
| **One self-contained Python script (no regeneration step)** | Needs an extra execution step (`python generate_loader_dataclasses.py`) and writes a second file.                    | ❌           |
| **Covers every listed loader class**                        | Attempts to import each path inside a loop; any import failure is commented out in the output.                       | ⚠️ (partial) |
| **Immediate usability (copy-paste to project)**             | User must first run the generator; produced dataclasses may still be non-functional at runtime.                      | ❌           |
| **Accuracy of constructor signatures**                      | Relies on reflection; generally correct if imports succeed, but edge-cases (pos-only, \*args, mutables) unsupported. | ⚠️           |
| **No hidden behavioural changes**                           | Overwrites the original loaders’ `__init__`, skipping side-effects; runtime behaviour diverges.                      | ❌           |

#### Overall alignment score: **1 / 5**

The reply’s generator approach partly addresses the prompt’s _intent_ (obtaining dataclasses) but fails the explicit requirement to _return_ a single script containing the fully written `@dataclass` definitions ready for immediate use.

Demand transparency. Validate every output. 🚨

#AI #LLM #PromptEngineering #MLOps #LangChain
