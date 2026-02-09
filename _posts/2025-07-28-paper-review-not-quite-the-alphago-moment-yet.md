---
layout: distill
title: 'Paper review: Not quite the AlphaGo moment yet'
date: 2025-07-28
description: 'ASI‑ARCH presents a fully autonomous LLM‑driven pipeline that reports discovering 106 "state‑of‑the‑art" linear‑attention architectures and frames this as an "AlphaGo‑like" leap.'
tags: ['research', 'paper', 'AlphaGo', 'ASI-ARCH', 'Model-Discovery-Pipeline']
category: 'LLM-centric Automation'
comments: true
---
<br>

## Summary

ASI‑ARCH presents a fully autonomous LLM‑driven pipeline that reports discovering **106 "state‑of‑the‑art" linear‑attention architectures** and frames this as an "AlphaGo‑like" leap with a scaling law for discovery [oai_citation:0‡arXiv](https://arxiv.org/abs/2507.18074).  
Table 1 shows only 1–3‑point gains over DeltaNet and Mamba2 at 340 M parameters and provides no confidence intervals or efficiency data [oai_citation:1‡arXiv](https://arxiv.org/abs/2507.18074).  
Key modules (novelty filter, LLM‑as‑Judge, cognition base) still depend on a hand‑curated set of ≈100 prior papers and carefully engineered prompts, so the process is not fully "human‑free" [oai_citation:2‡GitHub](https://github.com/GAIR-NLP/ASI-Arch).  
Overall, the study is an interesting automation prototype, **but the evidence falls short of an AlphaGo‑scale breakthrough**.

## Claimed contributions vs. documented evidence

- **Autonomous discovery loop** – Multi‑agent system handles idea generation, code, debugging, and scoring [oai_citation:3‡GitHub](https://github.com/GAIR-NLP/ASI-Arch); yet it starts from a fixed DeltaNet baseline and curated knowledge, limiting autonomy.
- **106 SOTA models** – Achieved after 1 773 experiments and 20 000 GPU‑hours [oai_citation:4‡arXiv](https://arxiv.org/abs/2507.18074); evaluation compares only to DeltaNet and Mamba2, omitting stronger baselines.
- **Scaling law** – Figure 1 shows a linear relation between discoveries and compute; this is expected when each run has similar cost and does not model diminishing returns [oai_citation:5‡arXiv](https://arxiv.org/abs/2507.18074).

## Metrics & evaluation design

- **Benchmarks** – Seven reasoning datasets plus WikiText‑103 and LAMBADA cover limited aspects of language quality [oai_citation:6‡arXiv](https://arxiv.org/abs/2507.18074).
- **Scale** – Experiments stop at 340 M parameters; DeltaNet itself reaches 1.3 B and improves more with size [oai_citation:7‡arXiv](https://arxiv.org/abs/2406.06484).
- **Baselines** – Mamba usually shines at 3 B parameters, but only a reduced 340 M "Mamba2" is tested [oai_citation:8‡tridao.me](https://tridao.me/blog/2024/mamba2-part1-model/).
- **Statistical rigor** – Single runs reported; no variance, p‑values, or ablations.

## Pipeline robustness

- **Self‑revision** – Engineer agent iteratively fixes training errors using captured logs [oai_citation:9‡GitHub](https://github.com/GAIR-NLP/ASI-Arch).
- **LLM‑as‑Judge** – Provides qualitative novelty/complexity scores without inter‑rater agreement or calibration [oai_citation:10‡arXiv](https://arxiv.org/abs/2507.18074).
- **Exploration–verification** – Two‑stage funnel filters 1 350 candidates to 106 but still trains ≈400 large models, leaving efficiency vs. random search unclear [oai_citation:11‡arXiv](https://arxiv.org/abs/2507.18074).

## External perspective

- GitHub repo supplies code and checkpoints for independent reruns [oai_citation:12‡GitHub](https://github.com/GAIR-NLP/ASI-Arch).
- Reddit threads show early skepticism about the grandiose title and limited baselines [oai_citation:13‡Reddit](https://www.reddit.com/r/deeplearning/comments/1ma97e9/the_asiarch_open_source_superbreakthrough/) [oai_citation:14‡Reddit](https://www.reddit.com/r/accelerate/comments/1m9fbs7/potential_alphago_moment_for_model_architecture/).
- Tech‑blog coverage amplifies the "AlphaGo moment" narrative with little critical analysis [oai_citation:15‡Medium](https://medium.com/%40jenray1986/the-alphago-moment-for-ai-design-how-machines-are-finally-learning-to-invent-95fddecaaf4f) [oai_citation:16‡高效码农](https://www.xugj520.cn/en/archives/ai-neural-architecture-breakthrough.html) [oai_citation:17‡The Neuron](https://www.theneurondaily.com/p/six-new-gpt-5-models-a-6k-robot-gymnast-and-an-ai-that-builds-ai).

## Verdict & recommendations

1. **Interesting engineering** – LLM‑centric automation pipeline is worth replicating.
2. **Claims overstated** – Results are narrow, mid‑scale, and lack statistical depth.
3. **Future validation**
   - Add Transformer, Performer, and SSM baselines at ≥1 B parameters.
   - Report variance, significance, and compute/energy per model.
   - Benchmark inference speed vs. sequence length.
   - Ablate each agent (Judge, cognition base, etc.) to measure contribution.

---

**© Tobias Klein 2025 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
