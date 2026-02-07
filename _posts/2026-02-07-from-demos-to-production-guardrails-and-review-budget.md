---
layout: distill
title: 'From Demos to Production: Guardrails and Review Budget'
date: 2026-02-07
description: "Part 3/3 | Generators output code fast. Your bottleneck is review budget - the amount of code you can verify without losing confidence. Optimize for reviewability, not throughput. Turn architecture preferences into enforceable constraints."
tags: ['llm', 'guardrails', 'code-review', 'automation', 'best-practices']
category: 'LLM-centric Automation'
comments: true
---
<br>

# From Demos to Production: Guardrails and Review Budget

### Part 3/3 - Designing Workflows Where Agents Stay Useful

This article is not about BERT or embedding models. It is about tool-using code agents built around an LLM: **code LLM agents (e.g., Claude Code, OpenAI Codex-style agents)**.

Two terms are used consistently:

- **Generator**: the chat/completion LLM the user interacts with, producing tokens under fixed **trained weights and biases** (where "bias" is the constant term in learned parameters, not a social/fairness concept).
- **Retriever/tools**: separate components that can be present or absent depending on the agent (repo search, file reading, running tests, linters, and other tooling).

With that clarified, the most practical question is not "how do I make the generator obey?", but "how do I set up a workflow where the generator's output is easy to verify and difficult to merge when it is wrong?"

## The bottleneck is usually human verification, not generation speed

Modern agents can generate large amounts of code quickly. That can feel productive in the moment, but it often shifts the cost to the engineer: you receive a large patch that you must now understand, review, and maintain.

In production engineering, your scarcest resource is frequently the review budget: the amount of code you can verify properly without losing confidence in the system. When an agent produces hundreds of lines across multiple files, the review budget is exceeded quickly. At that point, risk rises-not because the code is necessarily incorrect, but because it is insufficiently validated.

A good agent workflow therefore optimises for **reviewability**, not raw throughput.

## Prefer small, constrained changes over "complete modules"

A generator can be very helpful when you constrain the scope tightly:

- "Make a minimal change to route this parameter through the existing config object."
- "Refactor only this function to remove duplication; do not change call sites."
- "Add one validation check and one unit test that fails without it."

This keeps the patch size within your review budget and reduces the chance of architectural drift. It also allows the model to operate in a narrow region of the code, where local context is more reliable.

## Turn preferences into enforceable constraints

Many teams attempt to encode architecture in prompts: "use Hydra instantiation", "no ad hoc dependencies", "keep layers clean". Prompts help, but they are not enforcement.

Enforcement comes from checks that are:

- objective,
- repeatable,
- and automatic.

In practice, this means moving architectural expectations into mechanisms such as:

1. **Pre-commit and CI gates**
   Formatting, type checking, linting, and basic structural rules should fail fast.

2. **Dependency direction rules**
   Automated checks for forbidden imports and illegal dependencies prevent the most damaging forms of drift.

3. **Configuration schema validation**
   If configuration structure matters, validate it. Enforce required fields, allowed overrides, and invariants that reflect your architecture.

4. **Meaningful tests, not merely passing tests**
   Generators may produce tests that execute but assert little. Counter this by focusing on tests that encode behaviour: invariants, contracts, and failure modes that matter operationally.

The key idea is simple: the generator should not be able to "talk its way" into success. It should have to satisfy the same mechanical standards as a human contribution.

## Make the agent prove work with tool outputs

A reliable pattern is to require grounding:

- When the agent claims a fix, require the relevant test output, lint output, or error trace that demonstrates the improvement.
- When the agent proposes a refactor, require that a specific set of checks still pass.
- When a change touches configuration, require validation outputs or a minimal runtime check.

This moves the interaction from narrative to evidence. It also reduces the likelihood of the generator producing a confident explanation that is not anchored in actual execution.

## Address the "optimised for speed" bias directly

Many code agents are tuned to move quickly: generate patches, run tools, propose next steps. That is valuable, but it can also encourage overly large edits and "complete solutions" that look tidy yet exceed the review budget.

You can counterbalance this by setting explicit constraints in the workflow:

- "No new files unless essential."
- "Maximum patch size for a single iteration."
- "One behaviour change per patch."
- "No reformatting outside touched lines."

These are not stylistic preferences; they are operational controls that keep the agent useful in production.

## A calm mental model: proposer, verifier, and gatekeeper

A sustainable way to integrate code LLM agents into serious engineering work is to separate roles:

- The **generator** is a proposer. It can accelerate drafting, refactoring, and hypothesis generation.
- The **tools** provide evidence. They are your connection to the actual state of the system.
- The **guardrails** are the gatekeeper. They enforce architecture mechanically.

When those roles are clear, the working relationship becomes productive and less emotionally charged. You are no longer hoping that the generator "understands your architecture" in the human sense. You are designing a system where architecture is encoded in templates, constraints, and checks-so that even a fast, eager generator can contribute without quietly degrading the codebase.

That is often the difference between agent usage that feels impressive in demos and agent usage that remains dependable over months of production work.

## Links to the other posts in this series

[**From Demos to Production: Part 1**]({% link _posts/2026-02-05-from-demos-to-production-the-generator-is-not-learning-your-architecture.md %})<br>
[**From Demos to Production: Part 2**]({% link _posts/2026-02-06-from-demos-to-production-distribution-mismatch.md %})

---

**© Tobias Klein 2026 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
