---
layout: distill
title: 'From Demos to Production: Distribution Mismatch'
date: 2026-02-07
description: "Part 2/3 | LLM generators were trained on public code - simple examples, minimal composition. Your Hydra-driven, config-first architecture is far from that center. Under ambiguity, the model drifts toward common patterns, not your invariants."
tags: ['llm', 'hydra', 'config-driven', 'distribution-mismatch', 'mlops']
category: 'LLM-centric Automation'
comments: true
---
<br>

# From Demos to Production: Distribution Mismatch

### Part 2/3 - Why Public Patterns Don't Map to Your Hydra-Driven System

This article is not about BERT or embedding models. It is about tool-using code agents built around an LLM: **code LLM agents (e.g., Claude Code, OpenAI Codex-style agents)**. These systems wrap an LLM in an orchestration layer that can optionally retrieve code, read files, and run tools.

Two terms will be used precisely:

- **Generator**: the chat/completion LLM the user interacts with. Its behaviour is driven by its fixed **trained weights and biases** (with "bias" meaning the constant term in learned parameters, not a social/fairness concept) and by whatever context it receives.
- **Retriever/tools**: separate components that may add repo snippets, docs, logs, and command outputs to the generator's context.

With that established, the most common source of disappointment in "architectural prompting" is a form of distribution mismatch.

## Public code patterns are not production architecture patterns

Many generators were trained on a mixture of sources that strongly reflect what is publicly available and frequently represented. Regardless of the exact mix, the practical consequence is that a generator's defaults tend to align with patterns that are common, simple, and widely visible.

This becomes very tangible in systems that rely on sophisticated configuration and composition, such as Hydra-style workflows. "Config-driven" in production often means:

- many config groups,
- composition rules that encode product constraints,
- structured configuration objects,
- consistent instantiation paths,
- and strict separation between "what is configured" and "what is constructed".

In public repositories, the same tooling is often used in a simplified way: small examples, minimal composition, and conventions that are not enforced across a large platform.

If your codebase uses configuration as an architectural backbone rather than as a convenience layer, you are asking the generator to operate far away from the centre of what it most frequently sees. Under ambiguity, it will often produce code that is plausible in isolation but inconsistent with your architecture.

## Why documentation alone rarely closes the gap

A common expectation is that "if the agent can read the docs, it will know how to apply them." In practice, documentation tends to describe feature surfaces: what is possible, what flags exist, what decorators do. It rarely shows the organisational patterns that make the tool effective at scale.

Production usage patterns are often:

- organisation-specific,
- dependent on surrounding tooling,
- and anchored in conventions that are enforced socially and mechanically.

This is not a critique of documentation; it is simply a reality of complex systems. The "how" of a mature architecture is often encoded in internal examples, templates, and guardrails-artefacts that public docs cannot fully replicate.

## What the generator is actually doing with your inputs

Even with a tool-enabled agent, the generator is still generating tokens under its trained weights and biases, conditioned on the text it sees. It will try to compress what it reads into a few internal cues:

- which symbols appear important,
- which imports suggest a pattern,
- which files look "central",
- which lines resemble familiar templates.

When the visible code is large, interconnected, and shaped by conventions, those cues can be incomplete or misleading. The generator may correctly identify the names of relevant functions and still miss the invariants that matter. The result is a patch that looks coherent yet subtly changes the architecture: a direct instantiation appears where only config-driven instantiation is acceptable; a dependency boundary is crossed; a convenience shortcut is introduced.

## Why "just list the files" is not sufficient

Providing a manifest of involved files is helpful, but it does not automatically provide the semantics the generator needs. In architecture-heavy systems, correctness depends on relationships:

- which module is allowed to call which constructor,
- which config group owns a parameter,
- how overrides are resolved,
- which object graph is assembled where.

A list of files improves coverage; it does not guarantee comprehension. In many production settings, "understanding" is not just reading code; it is understanding the design intent that sits behind the code.

## A constructive way to reduce distribution mismatch

If the generator's priors do not match your architecture, the most practical response is to feed it a better local distribution. That typically means creating artefacts that make your patterns easy to retrieve and hard to misinterpret:

1. **Canonical reference implementations**
   A small number of "golden path" modules that demonstrate your intended patterns end-to-end: configuration layout, instantiation, logging, validation, and error handling.

2. **Templates that encode architecture**
   Internal scaffolds that make it difficult to do the wrong thing. If a new component must be config-driven, provide a template that begins config-first and leaves little room for ad hoc construction.

3. **Architecture rules as checkable constraints**
   Dependency direction checks, forbidden import rules, and structural validations make the pattern enforceable instead of aspirational.

4. **Tool outputs as grounding material**
   Logs, stack traces, and real command outputs help the agent remain anchored. When the generator must account for concrete evidence, it is less likely to drift into plausible-but-wrong narratives.

## When to use the agent, and when to keep it as support

A useful dividing line is the type of work:

- For **diagnostics**, triage, and reading large logs, code LLM agents are often excellent. They help you search the hypothesis space quickly.
- For **architecture-sensitive implementation across many files**, they can still contribute, but typically best as a collaborator: proposing small, reviewable patches, assisting with refactors one step at a time, and explaining how to reduce risk-rather than producing large, sweeping changes in one go.

The more your system depends on conventions that are not widely represented in public code, the more important it becomes to treat the agent as a fast proposer and rely on your own architecture artefacts and checks for correctness.

## Links to the other posts in this series

[**From Demos to Production: Part 1**]({% link _posts/2026-02-07-from-demos-to-production-the-generator-is-not-learning-your-architecture.md %})<br>
[**From Demos to Production: Part 3**]({% link _posts/2026-02-17-from-demos-to-production-guardrails-and-review-budget.md %})

---

**© Tobias Klein 2026 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
