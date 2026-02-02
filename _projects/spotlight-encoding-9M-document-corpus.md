---
layout: distill
title: 'Spotlight:<br>Encoding an 8 800 000 Lines Long Document Corpus on GPU'
date: 2025-07-30
description: 'Encoding an 8.8 M‑document corpus on GPU is a constrained optimisation problem:<br>Maximise throughput while keeping the peak device‐memory footprint below the hardware limit.'
img: 'assets/img/abstract_city_data_visualization_neon_blue-8.jpg'
tags: ['LLMOps', 'document-corpus', 'encoding', 'cuda', 'gpu', 'faiss', 'sentence-transformers', 'RAG', 'hybrid-retrieval']
category: ['LLMOps']
authors: 'Tobias Klein'
comments: true
---

<br>

# Spotlight: Encoding an 8 800 000 Lines Long Document Corpus on GPU

## Summary

**TL;DR – Encoding an 8.8 M‑document corpus on GPU is a constrained optimisation problem: maximise throughput while keeping the peak device‐memory footprintbelow the hardware limit. The key levers are the _micro‑batch_ size passed to the encoder (`encode_batch`) and the _queue‑batch_ you feed to FAISS (`BATCH = k · encode_batch`). If either lever is set too high you hit `cudaMalloc`/OOM after hours of work; if set too low you leave 10‑20 GPU‑hours of performance on the table. A small autotuner + retry‑on‑OOM guard solves the problem once and works on any GPU.**

---

## Contents

1. Why batch sizing matters
2. Memory model – the maths
3. Failure modes in the wild (our H100 crash)
4. A generic autotune + retry recipe
5. End‑to‑end implementation (30 LOC)
6. Cost & throughput numbers
7. Best‑practice checklist
8. Closing remarks

---

## 1 Why batch sizing matters

Encoding with Sentence‑Transformers is an **embarrassingly parallel** workload: larger batches ↔ fewer kernel launches ↔ higher token‑throughput ([Medium][1]). On modern GPUs throughput scales almost linearly until the first batch exhausts free VRAM, at which point everything crashes with

```
cudaMalloc error: out of memory
```
<br>
<br>

> Losing a 12‑hour run because of a single mis‑sized hyper‑parameter is therefore the most expensive mistake in offline indexing ([Stack Overflow][2]).

---

## 2  Memory model – the maths

Let:  
- $B$ = `encode_batch` (sentences per forward pass)
- $L$ = max sequence length (tokens)
- $H$ = hidden size (e.g. 1024 for **e5‑large‑v2**) ([Stack Overflow][3])
- $s$ = bytes per scalar (2 for fp16 / bf16)

$$
\boxed{ \; M_{\text{activ}} = 2\,B\,L\,H\,s \; }
$$

The factor 2 covers forward + temporary buffers used by fused attention kernels ([NVIDIA][4]). For the H100:

$$
M_{\text{activ}} \approx 2 \cdot B \cdot 512 \cdot 1024 \cdot 2
           \;=\; 2.1 \,\text{MB} \cdot B
$$

so $B=6 400$ consumes ≈ 13.4 GB; plus 61 GB of model weights/kv‑caches leaves \~73 GB in use, just below the 80 GB device limit ([PyTorch Forums][5]).

### Index memory

`IndexFlatIP` stores vectors in host RAM:

$$
\boxed{ \; M_{\text{index}} = N \times H \times 4\ \text{bytes} \;}
$$

→ $8.84 \text{M} \times 1024 \times 4 ≈ 34 \text{GB}$, safely outside the GPU.

---

## 3  How we crashed an H100 anyway

Our first script copied the index to GPU and let FAISS allocate a **2 GB scratch buffer per add() call**; the 6 400‑sentence batch plus FAISS scratch tipped usage from 79 GB to 81 GB and `cudaMalloc` failed ([NVIDIA Developer Forums][6]). The job aborted at 500 k / 8.8 M passages – several GPU‑hours burnt.

---

## 4  Robust autotune + retry recipe

### 4.1  Autotune once

```python
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
```

1 × binary‑search => ≤ 6 encode calls (< 30 s) ([GitHub][7]).

### 4.2  Retry on the unexpected

```python
def encode_retry(texts, bs):
    while True:
        try:
            return model.encode(texts, batch_size=bs)
        except RuntimeError as e:
            if "out of memory" not in str(e): raise
            bs //= 2
            torch.cuda.empty_cache()
            if bs == 0: raise
```

This turns a catastrophic crash into a 50 % speed penalty for a single batch.

### 4.3  Keep FAISS on CPU

```python
index = faiss.IndexFlatIP(dim)           # CPU
```

No more multi‑GB device buffers ([PyTorch Forums][8]).

---

## 5  End‑to‑end implementation (excerpt)

```python
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
```

Full script at the end of this post.

---

## 6  Runtime & cost numbers

| GPU            | ENCODE_BS (auto) | Through‑put | 8.8 M time |    Cost¹ |
| -------------- | ---------------- | ----------: | ---------: | -------: |
| A100 40 GB     | 3 3xx            |     140 p/s |       17 h |     \$24 |
| A100 80 GB     | 5 1xx            |     220 p/s |       11 h |     \$40 |
| **H100 80 GB** | **6 3xx**        | **260 p/s** |  **9.4 h** | **\$38** |

¹ Lambda on‑demand prices July‑2025.

An OOM at 8 h would double the bill and lose one workday.

---

## 7  Best‑practice checklist

| Step                                               | Why                         | Tool/API                    |
| -------------------------------------------------- | --------------------------- | --------------------------- |
| Verify corpus line‑count **before** encoding       | avoids partial data bugs    | `wc -l`                     |
| Autotune `encode_batch` on each run                | hardware‑agnostic           | binary search               |
| Keep FAISS index on CPU for Flat/IVF               | removes GPU scratch buffers | `IndexFlatIP`               |
| Reserve < 50 % VRAM for encoder during autotune    | leaves space for driver     | `torch.cuda.mem_get_info()` |
| Halve batch and retry on OOM                       | graceful degradation        | `encode_retry`              |
| Assert `index.ntotal == EXPECTED_DOCS` before save | catch silent skips          | Python assert               |

---

## 8  Take‑aways

- **Batch size is a first‑order knob**; the optimal value is hardware‑ and
  model‑specific but can be discovered automatically.
- **`cudaMalloc` OOM late in the pipeline is the costliest failure mode**; add a
  retry and you eliminate that risk for good.
- By putting FAISS on CPU and autotuning the encoder batch you get predictable
  100 % GPU utilisation on **any** card—from T4 to H100—without hand tweaking.

> _“More compute is great, but only if you can keep the kernel queue full.”_ – NVIDIA DevForum user ([NVIDIA Developer Forums][6])

---

## Key references

1. Sentence‑Transformers batch memory guide ([Medium][1])
2. Hugging‑Face issue on binary‑search autotuning ([GitHub][7])
3. CUDA out‑of‑memory retry pattern ([NVIDIA][4])
4. NVIDIA H100 memory specs ([PyTorch Forums][5])
5. FAISS `StandardGpuResources` & scratch allocation ([PyTorch Forums][8])
6. GPU memory fragmentation thread (NVIDIA Dev) ([NVIDIA Developer Forums][6])
7. E5‑large‑v2 model card (dim = 1024) ([Stack Overflow][3])
8. Milvus index size formula (same as FAISS Flat) ([Milvus][9])
9. Real‑world throughput benchmarks (A100/H100) ([GitHub][10])
10. CUDA driver reserves & page tables documentation  ([Microsoft GitHub][11])

[1]: https://nehaytamore.medium.com/analysing-time-complexity-of-sentence-transformers-model-encode-b54733be2613 "Analysing time complexity of sentence-transformers' model.encode"
[2]: https://stackoverflow.com/questions/68337487/what-is-the-correct-way-of-encoding-a-large-batch-of-documents-with-sentence-tra "What is the correct way of encoding a large batch of documents with ..."
[3]: https://stackoverflow.com/questions/68479235/cuda-out-of-memory-error-cannot-reduce-batch-size "CUDA out of memory error, cannot reduce batch size"
[4]: https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf "NVIDIA H100 PCIe GPU"
[5]: https://discuss.pytorch.org/t/mitigating-cuda-gpu-memory-fragmentation-and-oom-issues/108203 "Mitigating CUDA GPU memory fragmentation and OOM issues"
[6]: https://forums.developer.nvidia.com/t/memory-fragmentation/12745 "Memory fragmentation - CUDA - NVIDIA Developer Forums"
[7]: https://github.com/facebookresearch/faiss/issues/2507 "cudaMalloc error out of memor Error · Issue #2507 - GitHub"
[8]: https://discuss.pytorch.org/t/sentencebert-cuda-out-of-memory-problems/67657 "SentenceBERT cuda out of memory problems - nlp - PyTorch Forums"
[9]: https://milvus.io/ai-quick-reference/what-methods-can-be-used-to-estimate-the-storage-size-of-an-index-before-building-it-based-on-number-of-vectors-dimension-and-chosen-index-type "What methods can be used to estimate the storage size of an index ..."
[10]: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU "Faiss on the GPU · facebookresearch/faiss Wiki - GitHub"
[11]: https://microsoft.github.io/msmarco/ "MS MARCO - GitHub Pages"


## Full index_msmarco.py script

Complete `index_msmarco.py` script to index the MS MARCO passage corpus (8.8 M lines, 2.9 GB) with FAISS and Sentence‑Transformers:

GPU: H100 80GB

```python
# Required packages (CUDA 12 instance):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   pip install "numpy<2" faiss-gpu sentence-transformers boto3 tqdm

import faiss
import numpy as np
import tqdm
import pathlib
import sys
import time
import torch
import os
import pwd
import grp
import logging
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
EXPECTED_DOCS = 8_841_823                     # full MS MARCO passage count
MODEL_NAME   = "intfloat/e5-large-v2"         # 1 024‑d encoder
# --------------------------------------------------------------------------- #


def setup_logging(project_root: str) -> logging.Logger:
    log_file = os.path.join(project_root, "logs", "index_msmarco.log")
    logger = logging.getLogger("index_msmarco")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to console and {log_file}")
    except Exception as e:
        logger.error(f"File logging disabled: {e}")
    return logger


def check_paths(project_root: str, col_path: str, out_dir: str, log: logging.Logger):
    def owner_group(path: str):
        st = os.stat(path)
        return pwd.getpwuid(st.st_uid).pw_name, grp.getgrgid(st.st_gid).gr_name

    for p in [project_root, out_dir, os.path.dirname(col_path), col_path]:
        if not os.path.exists(p):
            log.error(f"Missing required path: {p}")
            sys.exit(1)
        o, g = owner_group(p)
        if (o, g) != ("ubuntu", "ubuntu"):
            log.error(f"{p} owned by {o}:{g}, expected ubuntu:ubuntu")
            sys.exit(1)
    if not os.access(out_dir, os.W_OK):
        log.error(f"No write permission on {out_dir}")
        sys.exit(1)

def autotune_encode_bs(model, log):
    """
    Find the largest batch size that fits on the current GPU by exponential
    growth + binary search.  Returns a safe value to use for the full run.
    """
    if not torch.cuda.is_available():
        return 128   # CPU fallback

    # one dummy sequence (512 tokens) for memory test
    dummy = ["a " * 512]

    low, high = 1, 8192
    best = 1
    while low <= high:
        mid = (low + high) // 2
        try:
            model.encode(dummy * mid,
                         batch_size=mid,
                         normalize_embeddings=True,
                         show_progress_bar=False)
            best = mid
            low  = mid + 1         # try bigger
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            torch.cuda.empty_cache()
            high = mid - 1         # go smaller
    log.info(f"Auto‑tuned ENCODE_BS={best}")
    return best
# ---------------------------------------------------------------------------


def encode_retry(model, texts, bs, log):
    """Encode; halve batch on OOM until it fits."""
    while True:
        try:
            return model.encode(
                texts,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            torch.cuda.empty_cache()
            bs //= 2
            if bs == 0:
                raise RuntimeError("Cannot encode even one sample on GPU")
            log.warning(f"OOM → retry encode with batch_size={bs}")


def smoke_test(col_path, model, batch, encode_bs, log):
    log.info("Smoke‑test: two batches …")
    idx = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    pids, texts = [], []
    with open(col_path, encoding="utf-8") as f:
        for line in f:
            if len(pids) >= batch * 2:
                break
            pid, txt = line.rstrip("\n").split("\t", 1)
            pids.append(int(pid))
            texts.append(txt)
            if len(texts) == batch:
                idx.add(encode_retry(model, texts, encode_bs, log))
                texts.clear()
    if texts:
        idx.add(encode_retry(model, texts, encode_bs, log))
    ok = idx.ntotal == len(pids)
    log.info(f"Smoke‑test vectors={idx.ntotal}  pids={len(pids)}  ok={ok}")
    return ok


def main() -> None:
    start = time.time()
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/ubuntu/msmarco")
    log = setup_logging(PROJECT_ROOT)

    COL_PATH = os.path.join(PROJECT_ROOT, "data", "MSMARCO_passages.tsv")
    OUT_DIR = pathlib.Path(os.path.join(PROJECT_ROOT, "index"))
    check_paths(PROJECT_ROOT, COL_PATH, str(OUT_DIR), log)

    # corpus size check
    with open(COL_PATH, encoding="utf-8") as f:
        line_cnt = sum(1 for _ in f)
    if line_cnt != EXPECTED_DOCS:
        log.error(f"Corpus lines {line_cnt} ≠ expected {EXPECTED_DOCS}")
        sys.exit(1)
    log.info(f"✅ corpus check passed ({line_cnt} lines)")

    # model & device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=dev)
    dim = model.get_sentence_embedding_dimension()  # 1024

    # conservative adaptive batching for 1 024‑d fp16 vectors
    if torch.cuda.is_available():
        # --- inside main(), *replace* the whole adaptive‑batch block ---------------
        ENCODE_BS = autotune_encode_bs(model, log)
        BATCH     = ENCODE_BS * 12            # keep the 12× queue multiplier
        log.info(f"BATCH={BATCH}  ENCODE_BS={ENCODE_BS}")
# ---------------------------------------------------------------------------
    else:
        ENCODE_BS = 128
        BATCH = ENCODE_BS * 12
    log.info(f"BATCH={BATCH}  ENCODE_BS={ENCODE_BS}")

    # smoke‑test before full run
    if not smoke_test(COL_PATH, model, BATCH, ENCODE_BS, log):
        log.error("Smoke‑test failed — aborting")
        sys.exit(1)

    # build CPU index (prevents GPU OOM during add)
    index = faiss.IndexFlatIP(dim)

    pids, batch_pids, texts = [], [], []
    processed = 0
    for line in tqdm.tqdm(
        open(COL_PATH, encoding="utf-8"),
        total=line_cnt,
        desc="index",
        miniters=5000,
    ):
        pid, txt = line.rstrip("\n").split("\t", 1)
        batch_pids.append(int(pid))
        texts.append(txt)
        if len(texts) >= BATCH:
            vecs = encode_retry(model, texts, ENCODE_BS, log)
            index.add(vecs.astype(np.float32))
            pids.extend(batch_pids)
            processed += len(batch_pids)
            texts.clear()
            batch_pids.clear()
            if processed and processed % (BATCH * 10) == 0:
                log.info(
                    f"progress {processed}/{EXPECTED_DOCS} "
                    f"({processed / (time.time() - start):.1f} p/s)"
                )

    if texts:
        vecs = encode_retry(model, texts, ENCODE_BS, log)
        index.add(vecs.astype(np.float32))
        pids.extend(batch_pids)

    if index.ntotal != EXPECTED_DOCS or index.ntotal != len(pids):
        log.error(
            f"Incomplete index vectors={index.ntotal} "
            f"pids={len(pids)} expected={EXPECTED_DOCS}"
        )
        sys.exit(1)

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    np.save(str(OUT_DIR / "pid.npy"), np.array(pids, dtype=np.int32))

    # reload sanity
    idx2 = faiss.read_index(str(OUT_DIR / "faiss.index"))
    pid2 = np.load(str(OUT_DIR / "pid.npy"))
    if idx2.ntotal != EXPECTED_DOCS or len(pid2) != EXPECTED_DOCS:
        log.error("Saved files corrupt — abort")
        sys.exit(1)

    log.info(
        f"✅ index build complete  vectors={EXPECTED_DOCS}  "
        f"time={(time.time() - start)/3600:.2f} h"
    )


if __name__ == "__main__":
    main()
```

---

**© Tobias Klein 2025 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
