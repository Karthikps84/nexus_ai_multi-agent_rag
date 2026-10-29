# RAG Evaluation Guide — Step by Step

This guide walks you through evaluating whether your Multi-Agent RAG system returns correct answers. Follow the steps in order.

---

## What we're measuring

| Layer | What it checks | Tool |
|---|---|---|
| Runtime | Latency, citation coverage, routing correctness, lexical answer overlap | Built-in (no extra deps) |
| RAGAS | Faithfulness, answer relevancy, context precision, context recall | `pip install ragas datasets` |
| DeepEval | Hallucination rate, contextual precision, answer relevancy | `pip install deepeval` |

---

## Prerequisites

```bash
# Core eval deps
pip install ragas datasets deepeval

# Confirm your .env has both keys
cat .env | grep -E "NVIDIA_API_KEY|SERPER_API_KEY"
```

---

## Step 1 — Index your documents (if not done yet)

Start the app and upload your files, or run directly:

```bash
python main.py
# (just starts the system, which triggers indexing if storage/ is empty)
# Ctrl-C once it prints "Index ready"
```

Verify the index built correctly:

```bash
ls storage/   # should contain chroma.sqlite3 and related files
```

---

## Step 2 — Build your eval set

This is the most important step. The eval set is a list of questions with **known correct answers** drawn from your actual documents. The builder generates them automatically using the LLM, but you **must review and correct** them before running the eval — an incorrect ground truth makes every metric meaningless.

```bash
# Generate 5 Q&A pairs per file (default)
python eval/build_eval_set.py

# More pairs, include hard reasoning questions
python eval/build_eval_set.py --per-file 8 --include-hard

# Target specific files only
python eval/build_eval_set.py --files resume.pdf paper.pdf --per-file 6
```

This writes `eval/eval_set.json`. Open it and review every entry:

```json
[
  {
    "question": "What university did Karthik attend for his Master's?",
    "ground_truth": "TU Kaiserslautern",
    "source_file": "resume.pdf",
    "page": 0,
    "difficulty": "easy"
  },
  ...
]
```

**Review checklist for each entry:**
- [ ] The ground truth is factually correct (check against the actual document)
- [ ] The question is answerable from the document alone
- [ ] The answer is specific enough to be verifiable (not vague like "it depends")
- [ ] Edit any wrong or vague ground truths directly in the JSON file

Aim for at least **10-15 questions** covering different parts of your corpus. More is better — 30+ gives stable metric estimates.

**Good ground truth examples:**
```json
"ground_truth": "TU Kaiserslautern, Germany"       ✓ specific
"ground_truth": "The paper proposes self-attention" ✓ specific
```

**Bad ground truth examples:**
```json
"ground_truth": "It discusses machine learning"    ✗ too vague to score
"ground_truth": "See the paper for details"        ✗ not a real answer
```

---

## Step 3 — Run the evaluation

```bash
# Full evaluation (all metric layers)
python eval/run_eval.py

# Skip slow RAGAS if you want fast runtime-only results first
python eval/run_eval.py --skip-ragas --skip-deepeval

# Evaluate only easy questions (good for a quick sanity check)
python eval/run_eval.py --filter easy --skip-deepeval

# Save output to a specific directory
python eval/run_eval.py --out eval/results_v1
```

Results are written to `eval/results/` as both a JSON file (full data) and a Markdown report (human-readable summary).

---

## Step 4 — Read the results

### Runtime metrics (always computed)

```
avg_latency_seconds     How long each question took end-to-end
p95_latency_seconds     Worst-case latency (the slow 5%)
avg_confidence          Verifier Agent's lexical overlap score (0–1)
citation_coverage       What % of answers had at least one cited source
local_routing_rate      What % of questions were answered from your docs
avg_lexical_f1          Token-overlap F1 between answer and ground truth
```

**Minimum acceptable thresholds for a healthy system:**

| Metric | Good | Acceptable | Fix needed |
|---|---|---|---|
| avg_latency_seconds | < 8s | 8–20s | > 20s |
| avg_confidence | > 0.55 | 0.35–0.55 | < 0.35 |
| citation_coverage | > 0.80 | 0.60–0.80 | < 0.60 |
| local_routing_rate | > 0.85 | 0.70–0.85 | < 0.70 |
| avg_lexical_f1 | > 0.35 | 0.20–0.35 | < 0.20 |

### RAGAS metrics (requires `pip install ragas datasets`)

| Metric | What it means | Good threshold |
|---|---|---|
| faithfulness | Are claims in the answer supported by retrieved context? | > 0.80 |
| answer_relevancy | Does the answer address the question? | > 0.75 |
| context_precision | Are the top retrieved chunks actually relevant? | > 0.70 |
| context_recall | Did retrieval find all the chunks needed to answer? | > 0.65 |

**The most important metric for correctness is faithfulness.** A low faithfulness score means the system is hallucinating — generating claims not supported by the retrieved text.

### DeepEval metrics

| Metric | What it means | Good threshold |
|---|---|---|
| hallucination | Rate of factual claims unsupported by context (lower is better) | < 0.20 |
| answer_relevancy | Same as RAGAS relevancy, different judge model | > 0.70 |
| contextual_precision | Ranking quality of retrieved chunks | > 0.65 |

---

## Step 5 — Diagnose problems

### Problem: Low faithfulness (< 0.60)

The system is hallucinating. Most likely cause: the retrieved chunks don't contain the answer, so the LLM fills in from its training data instead.

**Fixes to try in order:**
1. Lower `SIMILARITY_THRESHOLD` in `config.py` from 0.45 → 0.35 — makes it harder to fall through to web search prematurely.
2. Increase `SIMILARITY_TOP_K` from 8 → 12 — retrieves more candidates before reranking.
3. Increase `CHUNK_SIZE` from 900 → 1200 — chunks with more context are less likely to be orphaned facts.
4. Check if ENABLE_RERANK = True — reranking significantly improves which chunks are actually used.

### Problem: Low context_recall (< 0.50)

Retrieval is missing relevant chunks entirely.

**Fixes:**
1. Confirm `ENABLE_HYBRID_SEARCH = True` in config — BM25+dense fusion covers more ground than dense alone.
2. Reduce `CHUNK_OVERLAP` from 150 → 50 — less overlap means more unique coverage in fewer chunks.
3. Delete `storage/` and re-index with the current config — changes to CHUNK_SIZE only take effect on a full rebuild.

### Problem: Low local_routing_rate (< 0.70)

Too many questions fall through to web search instead of being answered from your documents.

**Fix:** The SIMILARITY_THRESHOLD is too high for your corpus and embedding model combination. Reduce it:

```python
# config.py
SIMILARITY_THRESHOLD = 0.38  # try lower values in 0.05 increments
```

Run `python eval/run_eval.py --skip-ragas --skip-deepeval` after each change to see if local_routing_rate improves.

### Problem: High latency (p95 > 30s)

**Diagnose which stage is slow** — look at the per-agent trace in the dev panel of the Streamlit UI. Common causes:

| If slow stage is... | Fix |
|---|---|
| embed+ANN search | Increase EMBED_BATCH_SIZE, check NIM endpoint health |
| Local generation | Switch from 70B to 8B for that route in MODEL_ROUTING config |
| Web search | Questions routing to web can't be made faster from code — improve local retrieval so they hit the KB instead |
| Reranking | Set ENABLE_RERANK = False as a test; if p95 drops, the reranker is the bottleneck |

### Problem: Low avg_lexical_f1 (< 0.15)

The answers don't share much vocabulary with the ground truths at all.

**Before tuning the system**, check if the ground truths themselves are wrong. Manually look at 5 failing pairs:

```bash
# Open the latest results JSON and look at the per-question answers
cat eval/results/results_*.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data['results'][:5]:
    print('Q:', r['question'])
    print('Expected:', r['ground_truth'])
    print('Got:', r['answer'][:200])
    print()
"
```

If the answers look correct but the ground truths are wrong, fix the eval set. If the answers are genuinely wrong, the problem is retrieval or generation quality.

---

## Step 6 — Iterative improvement loop

```
Edit config.py  →  rm -rf storage/  →  python main.py (rebuilds index)
→  python eval/run_eval.py --skip-deepeval
→  compare metrics to previous run
→  repeat
```

**One config change per iteration.** Changing CHUNK_SIZE and SIMILARITY_THRESHOLD at the same time makes it impossible to know which one helped or hurt.

**Track runs** — the eval runner timestamps its output files, so you can compare `results_20260806_120000.md` vs `results_20260806_130000.md` directly.

---

## Step 7 — Continuous eval (optional)

Once you have a stable eval set, add this to your workflow:

```bash
# Run before any significant code change as a regression check
python eval/run_eval.py --skip-deepeval --out eval/baseline

# After your change
python eval/run_eval.py --skip-deepeval --out eval/after_change

# Diff the markdown reports
diff eval/baseline/report_*.md eval/after_change/report_*.md
```

This catches regressions — cases where a "fix" to one metric accidentally breaks another.

---

## Quick reference

```bash
# Full pipeline from scratch
python eval/build_eval_set.py --per-file 6 --include-hard
# → review eval/eval_set.json manually
python eval/run_eval.py
# → read eval/results/report_*.md

# Fast sanity check (no RAGAS/DeepEval API calls)
python eval/run_eval.py --skip-ragas --skip-deepeval --filter easy

# Diagnose one specific file's retrieval quality
python eval/build_eval_set.py --files yourfile.pdf --per-file 10
python eval/run_eval.py --skip-ragas --skip-deepeval
```

---

## Vision Analysis Evaluation (real object detection)

Text RAG metrics (faithfulness, lexical F1) don't apply to object detection — this needs its own eval, using precision/recall/IoU against ground-truth bounding boxes, the same family of metrics used to evaluate any real detector (this is the same evaluation family your thesis used for YOLOv11/Mask R-CNN/SAM benchmarking).

### Quick smoke test (no labeling required)

```bash
python eval/eval_vision.py
```

Runs detection on every image in `data/`, reports detection count, average confidence, and latency per image. Good for "is it working at all" — not a correctness measure.

### Full evaluation (precision/recall/IoU)

1. Pick 5-10 images from your `data/` folder that have a clear, countable set of objects.
2. Draw ground-truth bounding boxes for each — any tool works (LabelImg, CVAT, Roboflow's own annotation UI, or just eyeball pixel coordinates by opening the image and reading position). Write them to `eval/vision_eval_set.json`:

```json
[
  {
    "image": "sample1.jpg",
    "ground_truth_boxes": [
      {"class": "person", "x1": 34, "y1": 12, "x2": 210, "y2": 400},
      {"class": "car",    "x1": 250, "y1": 100, "x2": 480, "y2": 300}
    ]
  }
]
```

3. Run:

```bash
python eval/eval_vision.py --annotated
```

### Reading the results

| Metric | What it means | Good threshold |
|---|---|---|
| Precision | Of everything detected, what fraction was actually correct | > 0.75 |
| Recall | Of everything that should've been detected, what fraction was found | > 0.70 |
| F1 | Harmonic mean of precision/recall | > 0.72 |
| Mean IoU (matches) | Box localization tightness on correct detections | > 0.60 |
| Mean absolute count error | Average |predicted - actual| object count | < 1.0 |

**If recall is low** (missing real objects): lower `VISION_ANALYSIS_CONFIDENCE` in `config.py` — the detector is filtering out real detections as too-low-confidence. Try 0.15 instead of 0.25.

**If precision is low** (lots of false positives): raise `VISION_ANALYSIS_CONFIDENCE`, or the scene has objects genuinely outside COCO's 80 trained classes (YOLO26's default weights are COCO-pretrained — domain-specific objects like microscopy particles need a custom-trained model, which is exactly the kind of extension a Roboflow-trained model would solve).

**If mean IoU is low but precision/recall are fine**: the detector is finding the right objects but boxes are loose — this is a model-scale issue (try `yolo26s.pt` instead of `yolo26n.pt`) rather than a threshold issue.

### The cross-check feature specifically

When you run "Detect + compare vs. VLM" in the UI, the Verifier Agent's `cross_check_vision()` compares the detector's count against any number mentioned in the vision-language model's description. To validate this is working correctly (not just present), test it directly:

```bash
python3 -c "
from agents.verifier import VerifierAgent
detection = {'success': True, 'model': 'yolo26n.pt', 'detections': [{'class':'person'}]*3}
note = VerifierAgent.cross_check_vision('I see 2 people in the photo.', detection)
print(note)  # should flag the 2 vs 3 mismatch
"
```

If it prints a mismatch note, the cross-check is working. This is your interview talking point: two independent methods (a generative description vs. a real detector's measurement) checking each other, and disagreeing when they should.
