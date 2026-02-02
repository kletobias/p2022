---
layout: distill
title: 'From Demos to Production: The Generator Is Not Learning Your Architecture'
date: 2026-02-07
description: "Inference doesn't rewrite trained weights. When you tell an LLM 'use config-driven instantiation,' you're adding a prompt constraint - not teaching it your architecture. That gap explains why real-world, multi-file work remains fragile."
tags: ['llm', 'code-agents', 'architecture', 'inference', 'machine-learning']
category: 'LLM-centric Automation'
comments: true
---
<br>

# From Demos to Production: The Generator Is Not Learning Your Architecture

## Part 1/3

It helps to start with clear boundaries, because the discussion around "AI coding" often blends together several different systems.

This article is not about BERT or embedding models. It is about tool-using code agents built around an LLM: **code LLM agents (e.g., Claude Code, OpenAI Codex-style agents)**. In these systems, an LLM is placed inside an orchestration loop that may (or may not) have access to tools such as file search, repo reading, test execution, linters, and command running.

Two terms matter throughout:

- **Generator**: the chat/completion LLM the user interacts with. It produces tokens based on its **trained weights and biases** (where "bias" is the constant term in the learned parameters, not a social/fairness concept) plus whatever context is provided at inference time.
- **Retriever/tools**: separate system components that can be present or absent depending on the agent. They influence what the generator sees, but they are not the generator itself.

With that framing, a core principle becomes easier to accept and work with:

### Inference does not rewrite trained weights and biases

In standard deployments, an LLM's trained weights and biases are fixed during inference. When you give an instruction like "use config-driven instantiation", you are not installing new behaviour into the model. You are adding a short-lived constraint into the prompt. The generator then tries to produce a continuation that is plausible under its trained weights and biases, given the prompt and any retrieved context.

This distinction explains a common gap between expectation and outcome. Many engineering instructions are not local syntax preferences; they are architectural commitments:

- where construction is allowed to happen,
- which layers may depend on which,
- how configuration groups compose,
- what "done" looks like from the standpoint of maintainability.

Those commitments are rarely explicit in one place. They emerge across dozens of files, conventions, and tooling. A generator can produce the right surface form of code and still miss the invariants that make the code "correct" for your system.

### Why real-world, multi-file work is still a weak fit

The challenge is not that a generator cannot ever output high-quality code. It is that, for architecture-sensitive changes spanning many files, reliability becomes the central issue.

In production settings it is common to have one subsystem change touching tens of files. Even if you provide a manifest (for example, a JSON list of dependencies and file paths), you are still asking the generator to do something quite specific:

1. build a consistent mental model across a wide surface area,
2. infer which patterns are incidental and which are invariants, and
3. apply changes without violating those invariants.

That is difficult for any system that primarily operates by generating likely continuations rather than by proving properties. It becomes even more difficult when the agent is optimised for speed, because speed optimisation typically means shorter deliberation per step and a stronger tendency to "move forward" with plausible edits.

This is why many experienced engineers adopt an operational heuristic: for large, multi-file architectural work, treat the agent as assistance-not as the primary implementer.

### "It feels like some models validate more than others"

In practice, different products exhibit different behaviours. Some interactive chat models appear more cautious, more willing to revise, and more inclined to provide intermediate reasoning. Many code agents appear tuned for throughput: they run tool loops quickly, generate patches rapidly, and keep the workflow moving.

It is important to describe this as an observed difference in product behaviour rather than as a claim about hidden internal mechanisms. The key point is practical: the agent's optimisation target influences your outcome. If the system is tuned to deliver output quickly, it will tend to generate more code per unit time and will depend more on you (and your checks) to catch subtle architectural drift.

### The "cheap confidence" problem

Most generators can produce fluent, confident text with very little friction. That does not mean they are consistently calibrated about whether a patch is correct in your environment. In fact, a common pattern is that the generator will provide a positive narrative ("this should work now", "tests pass") even when it has not executed anything, has partial context, or has quietly assumed missing details.

The issue is not that the model can never express uncertainty. It can. The issue is that uncertainty is not reliably produced unless you enforce it with your workflow. In other words: you often have to design the interaction such that the model cannot "succeed" without concrete evidence.

### Context window is not the same as comprehension

Long context windows help with retrieval and referencing, but they do not guarantee that the generator will preserve architectural invariants. Even when all relevant files fit into a large context window, the generator still has to allocate attention across a very large input, identify what matters, and remain consistent over long edits.

In production codebases, the limiting factor is frequently not "can the text fit", but "what does the model infer from what it sees", filtered through its trained weights and biases. When your architecture is unusual relative to what the model frequently saw during training, the generator's default completions will often drift toward more common patterns.

### A more productive stance

A helpful mental model is:

- The generator is a strong proposer of code and explanations.
- The system around it (retrieval, tools, and your checks) determines what is grounded in evidence.
- Your automation determines what is allowed to land.

When you shift from "teach the model my architecture" to "use the model where its strengths align, and enforce architecture mechanically", the experience becomes more predictable. The agent remains valuable, but you use it with clearer expectations: it is not learning your codebase; it is generating proposals under fixed trained weights and biases, guided by the context you provide and the constraints you enforce.

---
