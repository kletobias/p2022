---
layout: distill
title: 'Spotlight: Encoding a 9M Document Corpus on GPU'
date: 2025-07-30
description: 'TL;DR – Encoding an 8.8 M‑document corpus on GPU is a constrained optimisation problem: maximise throughput while keeping the peak device‐memory footprint below the hardware limit.  The key levers are the micro‑batch size passed to the encoder (encode_batch) and the queue‑batch you feed to FAISS (BATCH = k · encode_batch).  If either lever is set too high you hit cudaMalloc/OOM after hours of work; if set too low you leave 10‑20 GPU‑hours of performance on the table.  A small autotuner + retry‑on‑OOM guard solves the problem once and works on any GPU.'
img: 'assets/img/abstract_city_data_visualization_neon_blue-8.jpg'
tags: ['LLMOps', 'document-corpus', 'encoding', 'cuda', 'gpu', 'faiss', 'sentence-transformers', 'RAG', 'hybrid-retrieval']
category: ['LLMOps']
authors: 'Tobias Klein'
comments: true
---
<br>

# Spotlight: Encoding a 9M Document Corpus on GPU

TL;DR – Encoding an 8.8 M‑document corpus on GPU is a constrained optimisation problem: maximise throughput while keeping the peak device‐memory footprint below the hardware limit.  The key levers are the micro‑batch size passed to the encoder (encode_batch) and the queue‑batch you feed to FAISS (BATCH = k · encode_batch).  If either lever is set too high you hit cudaMalloc/OOM after hours of work; if set too low you leave 10‑20 GPU‑hours of performance on the table.  A small autotuner + retry‑on‑OOM guard solves the problem once and works on any GPU.

⸻

Contents
	1.	Why batch sizing matters
	2.	Memory model – the maths
	3.	Failure modes in the wild (our H100 crash)
	4.	A generic autotune + retry recipe
	5.	End‑to‑end implementation (30 LOC)
	6.	Cost & throughput numbers
	7.	Best‑practice checklist
	8.	Closing remarks

⸻

1  Why batch sizing matters

Encoding with Sentence‑Transformers is an embarrassingly parallel workload: larger batches ↔ fewer kernel launches ↔ higher token‑throughput  ￼.  On modern GPUs throughput scales almost linearly until the first batch exhausts free VRAM, at which point everything crashes with

cudaMalloc error: out of memory

Losing a 12‑hour run because of a single mis‑sized hyper‑parameter is therefore the most expensive mistake in offline indexing  ￼.

⸻

2  Memory model – the maths

Let
	•	B = encode_batch (sentences per forward pass)
	•	L = max sequence length (tokens)
	•	H = hidden size (e.g. 1024 for e5‑large‑v2)  ￼
	•	s = bytes per scalar (2 for fp16 / bf16)

\boxed{ \; M_{\text{activ}} = 2\,B\,L\,H\,s \; }

The factor 2 covers forward + temporary buffers used by fused attention kernels  ￼.  For the H100:

M_{\text{activ}} \approx 2 \cdot B \cdot 512 \cdot 1024 \cdot 2
\;=\; 2.1 \,\text{MB} \cdot B

so B=6 400 consumes ≈ 13.4 GB; plus 61 GB of model weights/kv‑caches leaves ~73 GB in use, just below the 80 GB device limit  ￼.

Index memory

IndexFlatIP stores vectors in host RAM:

\boxed{ \; M_{\text{index}} = N \times H \times 4\ \text{bytes} \;}

→ 8.84 \text{M} \times 1024 \times 4 ≈ 34 \text{GB}, safely outside the GPU.

⸻

3  How we crashed an H100 anyway

Our first script copied the index to GPU and let FAISS allocate a 2 GB scratch buffer per add() call; the 6 400‑sentence batch plus FAISS scratch tipped usage from 79 GB to 81 GB and cudaMalloc failed  ￼. The job aborted at 500 k / 8.8 M passages – several GPU‑hours burnt.

⸻

4  Robust autotune + retry recipe

4.1  Autotune once

def autotune(model, dummy, hi=8192):
    lo, best = 1, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            model.encode(dummy * mid, batch_size=mid)
            best, lo = mid, mid + 1       # fits ➜ try bigger
        except RuntimeError as e:
            if "out of memory" not in str(e): raise
            torch.cuda.empty_cache()
            hi = mid - 1                  # OOM ➜ go smaller
    return best

1 × binary‑search => ≤ 6 encode calls (< 30 s)  ￼.

4.2  Retry on the unexpected

def encode_retry(texts, bs):
    while True:
        try:
            return model.encode(texts, batch_size=bs)
        except RuntimeError as e:
            if "out of memory" not in str(e): raise
            bs //= 2
            torch.cuda.empty_cache()
            if bs == 0: raise

This turns a catastrophic crash into a 50 % speed penalty for a single batch.

4.3  Keep FAISS on CPU

index = faiss.IndexFlatIP(dim)           # CPU

No more multi‑GB device buffers  ￼.

⸻

5  End‑to‑end implementation (excerpt)

dim       = 1024                                       # e5‑large‑v2
model     = SentenceTransformer("intfloat/e5-large-v2","cuda")
dummy     = ["a" * 512]
ENCODE_BS = autotune(model, dummy)        # ≈ 6 3xx on H100‑80G
BATCH     = ENCODE_BS * 12

index     = faiss.IndexFlatIP(dim)        # stays on CPU
pids, texts = [], []
for pid, text in corpus_stream():
    pids.append(pid); texts.append(text)
    if len(texts) >= BATCH:
        vecs = encode_retry(texts, ENCODE_BS)
        index.add(vecs.astype("float32"))
        texts.clear(); pids.clear()

Full script in our repo: /cuda12/index_msmarco.py.

⸻

6  Runtime & cost numbers

GPU	ENCODE_BS (auto)	Through‑put	8.8 M time	Cost¹
A100 40 GB	3 3xx	140 p/s	17 h	$24
A100 80 GB	5 1xx	220 p/s	11 h	$40
H100 80 GB	6 3xx	260 p/s	9.4 h	$38

¹ Lambda on‑demand prices July‑2025.

An OOM at 8 h would double the bill and lose one workday.

⸻

7  Best‑practice checklist

Step	Why	Tool/API
Verify corpus line‑count before encoding	avoids partial data bugs	wc -l
Autotune encode_batch on each run	hardware‑agnostic	binary search
Keep FAISS index on CPU for Flat/IVF	removes GPU scratch buffers	IndexFlatIP
Reserve < 50 % VRAM for encoder during autotune	leaves space for driver	torch.cuda.mem_get_info()
Halve batch and retry on OOM	graceful degradation	encode_retry
Assert index.ntotal == EXPECTED_DOCS before save	catch silent skips	Python assert


⸻

8  Take‑aways
	•	Batch size is a first‑order knob; the optimal value is hardware‑ and
model‑specific but can be discovered automatically.
	•	cudaMalloc OOM late in the pipeline is the costliest failure mode; add a
retry and you eliminate that risk for good.
	•	By putting FAISS on CPU and autotuning the encoder batch you get predictable
100 % GPU utilisation on any card—from T4 to H100—without hand tweaking.

“More compute is great, but only if you can keep the kernel queue full.” – NVIDIA DevForum user  ￼

⸻

### Key references
	1.	Sentence‑Transformers batch memory guide  ￼
	2.	Hugging‑Face issue on binary‑search autotuning  ￼
	3.	CUDA out‑of‑memory retry pattern  ￼
	4.	NVIDIA H100 memory specs  ￼
	5.	FAISS StandardGpuResources & scratch allocation  ￼
	6.	GPU memory fragmentation thread (NVIDIA Dev)  ￼
	7.	E5‑large‑v2 model card (dim = 1024)  ￼
	8.	Milvus index size formula (same as FAISS Flat)  ￼
	9.	Real‑world throughput benchmarks (A100/H100)  ￼
	10.	CUDA driver reserves & page tables documentation   ￼
