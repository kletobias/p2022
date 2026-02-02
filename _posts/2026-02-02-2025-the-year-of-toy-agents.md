---
layout: distill
title: '2025: The Year of Toy Agents'
date: 2026-02-02
description: "Every vendor promised agents that 'just work.' Reality delivered toy agents - fragile in production, useless when correctness matters. AI doesn't remove complexity. It relocates it."
tags: ['llm', 'genai', 'ai-agents', 'governance', 'ai-in-business']
category: 'LLM-centric Automation'
comments: true
---
<br>

# 2025: The Year of Toy Agents

Today is 2 February 2026, and I'm looking back at a year that was supposed to be "the year of agents."

It wasn't.

It was the year of **agent demos**.

Everywhere you looked, vendors had the same storyline:
"Just use natural language. Our agent will write the query, fetch the data, do the work. No more technical barriers."

The screenshots were polished. The marketing decks were confident.
And in real-world environments? Most of these "agents" collapsed immediately under the weight of reality:

1. Ambiguous questions.
2. Messy schemas and half-documented systems.
3. Shifting business definitions.
4. Permissions, compliance, and auditability.
5. Cost and latency constraints.
6. The boring but non-negotiable requirement of being correct.

What shipped in 2025, in the vast majority of cases, were **toy agents**: safe enough for a demo, fragile in production, unusable when the cost of being wrong is non-trivial.

---

**I'm not looking at this from the outside**

Who am I to say this?

I'm not a marketing lead trying to hit an "AI" slide quota. I'm the person who:

- Pays for the strongest models out of my own pocket.
- Uses them daily on real systems where bugs, outages, and regressions matter.
- Treats AI less like magic and more like a stubborn, brilliant junior that has to be watched at all times.

Last year, I didn't "play" with AI. I **worked** with it.

Not in one-off chats or weekend experiments, but in long, messy, multi-day sessions, wired into terminals, codebases, CI, pre-commit hooks, and MLOps pipelines. I spent last year battling models, forcing them to be useful, and building the scaffolding to make their output safe enough to connect to reality.

And here is the core observation from that experience:

> AI doesn't remove complexity. It relocates it.
> The hard part isn't prompting. The hard part is **governance**.

---

**The fantasy: "AI makes it less technical"**

The current narrative is:
"AI will make everything less technical. Anyone can just tell the system what they want in plain language. The agent will figure it out."

That sounds empowering. It's also dangerously incomplete.

To let "anyone" safely control complex systems via natural language, you need someone who:

1. Deeply understands the system and its failure modes.
2. Anticipates the anti-patterns the model will generate.
3. Enforces hard constraints around what the agent can and cannot do.
4. Designs verification layers that don't trust the model just because the answer sounds confident.

That someone is not a casual user.
That someone is not a slide in a pitch deck.
That someone is a domain expert who is **actively governing** the AI.

Which leads to a very unpopular truth:
AI does not make things "less technical." It makes them **differently technical**.
The work moves from "write all the code yourself" to "design, constrain, and continuously audit a non-deterministic collaborator."

---

**What real AI work looked like for me in 2025**

When I say I "govern" AI, this is what that actually looks like in practice:

1. **Heavy, layered instructions**
   I don't rely on a single prompt or a pretty "system message."
   I use building blocks that define what the model is allowed to do, what it must never do, and how it should behave in specific contexts. These are not inspirational guidelines. They are constraints.

2. **Pre-commit hooks and guardrails**
   I don't just let an agent commit code.
   I wire it into a pipeline where its changes are checked, formatted, statically analysed, and tested before they get anywhere near main. The AI doesn't get to "be helpful" without being inspected.

3. **Pattern-spotting and anti-pattern hunting**
   Over time, you learn what your models consistently get wrong: subtle security issues, performance foot-guns, concurrency problems, data leakage, brittle abstractions.
   You start reading AI-generated output with a mental checklist of "usual crimes" and scanning for them constantly.

4. **Tight tool boundaries**
   Agents don't get a wide-open toolbox.
   They get very specific tools with well-defined contracts.
   They don't get to freely roam your systems. Everything is explicit, limited, and supervised.

5. **Verification first, trust never**
   The model's job is to propose; my job is to verify.
   I never confuse fluency with correctness. The more confident the output looks, the more suspicious I am.

That is what "harnessing" AI actually looks like in 2025.
Not "ask once, trust always," but "ask repeatedly, constrain aggressively, verify relentlessly."

---

**The mental cost no one likes to talk about**

The marketing story is: "AI makes things easier."

My experience: **it makes things faster, not easier.**

Before agents, work was slower but more linear. When you write everything yourself:

- You see every decision.
- You understand every trade-off.
- You feel the system's shape in your head.

With AI, the work changes:

1. You constantly read and review large volumes of generated content.
2. You look for invisible edge cases that are not obvious at a glance.
3. You reverse-engineer the model's choices and ask, "Why this? What did it miss?"
4. You run more tests, more often, because you don't trust the generator.

The result: your throughput goes up, but so does your cognitive load.

Ironically, it was **less** mentally taxing before agents existed.
You had fewer surprises, fewer hidden landmines, and more direct control.

Today, if you want to stay ahead, you don't really have the option of ignoring AI. But using it seriously means accepting that:

- Your brain will be fully engaged.
- You will spend a lot of time saying "no" to the model.
- Your value is not that you can type a clever prompt; your value is that you can **govern** an unruly but powerful system.

---

**Toy agents vs. real-world agents**

So when I see yet another announcement about:

> "Now you can query your database in natural language. Our agent will just handle it."

I don't see innovation. I see a risk disguised as a convenience.

Because in the real world:

1. Data has access boundaries and regulatory constraints.
2. Business logic lives in weird legacy corners nobody wrote down.
3. "Revenue" means one thing in Finance, another in Sales, and a third in Analytics.
4. A wrong answer isn't just "oops," it's a bad decision, a broken audit trail, or a compliance incident.

If there is no visible story about governance, constraints, observability, and evaluators, then it's not a production agent. It's a demo.

We don't need more toy agents in 2026.
We need agents that are:

1. Bounded.
2. Testable.
3. Auditable.
4. Operated by people who deeply understand the systems they're touching.

---

**Where this actually leaves us**

Despite all of this, I'm not pessimistic about AI.
Quite the opposite.

Last year showed me that:

- When you govern AI correctly, it is absurdly powerful.
- When you don't, it becomes a very efficient way to create expensive problems at scale.

The gap between those two outcomes is not the model.
It's the operator.

2025 was not the year where agents replaced experts.
It was the year where we started to understand that the only people who can reliably use agents are the ones who were already experts - and who are willing to stay fully engaged while the model does the typing.

If there's a mindset I've taken into 2026, it's this:

> Treat AI as a powerful but unpredictable collaborator.
> Don't worship it. Don't fear it.
> Govern it.

That's not as shiny as "everyone can be a developer now."
But it's honest. And in real-world systems, honest beats hype every single time.

---

**© Tobias Klein 2026 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
