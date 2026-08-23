```markdown
# RouterNet

A domain-specialized LoRA adapter suite and zero-overhead hard-routing engine for **Qwen3-4B-Instruct** — dispatches domain-specific prompts to isolated adapters (`creative`, `code`, `math`) without cross-task weight pollution or extra FLOP compute costs during generation.

## Repository Structure


```

RouterNet/
├── assets/                     # Architecture schematics & benchmark plots
├── notebooks/                  # End-to-end training & evaluation workflows
│   ├── 01-train-creative-adapter.ipynb
│   ├── 02-train-code-adapter.ipynb
│   ├── 03-train-math-adapter.ipynb
│   └── 04-comparative-benchmark.ipynb
├── adapters/                   # Trained PEFT/LoRA checkpoints
│   ├── creative/
│   ├── code/
│   └── math/
├── src/                        # Router & multi-adapter hot-swapping engine
│   ├── engine.py               # RouterNetEngine3Domain implementation
│   └── router_classifier.joblib# Serialized MiniLM-L6 router classifier
├── data/                       # Datasets & preprocessing outputs
└── docs/                       # System documentation

```

## Adapters

| Adapter | Domain | Datasets | LoRA rank ($r$) | LoRA alpha ($\alpha$) | LR |
|---|---|---|:---:|:---:|:---:|
| `creative` | Creative writing | [euclaise/WritingPrompts_curated](https://huggingface.co/datasets/euclaise/WritingPrompts_curated) | 16 | 32 | 5e-5 |
| `code` | Code generation | [nickrosh/Evol-Instruct-Code-80k-v1](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 16 | 32 | 5e-5 |
| `math` | Math reasoning | [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 32 | 64 | 2e-4 |

* **Base model:** [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
* **Execution precision:** Native FP16 (Single NVIDIA T4 GPU, ~8.2 GB total VRAM footprint)

## System Architecture

```text
                     ┌────────────────────────┐
                     │   Incoming User Query  │
                     └───────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │  SentenceTransformer Embedder │
                 │    (all-MiniLM-L6-v2)         │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │  Logistic Regression Router   │
                 └───────────────┬───────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
  [ Domain: MATH ]       [ Domain: CODE ]      [ Domain: CREATIVE ]
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │  model.set_adapter(selected)  │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │  Qwen3-4B Base Model (FP16)   │
                 └───────────────────────────────┘

```

## Empirical Benchmark Results

Evaluated across multi-step mathematical reasoning test suites (Algebra, Calculus, Number Theory, Probability, Linear Algebra):

| Evaluation Metric | Base Model (`Qwen3-4B-Instruct`) | RouterNet (`Base + Math Adapter`) | Performance Impact |
| --- | --- | --- | --- |
| **Math Accuracy** | 100% | **100%** | Maintained ground-truth accuracy |
| **CoT Proof Completion Rate** | 40% (Early Truncation) | **80%** | **+100% Completion** (via dynamic budgeting) |
| **Formatting Enforcement** | Generic Markdown | **Strict LaTeX (`\boxed{}`)** | Enforced dataset schema tags |
| **Cross-Task Weight Pollution** | High (Style Drift) | **Zero (Hard Isolation)** | Completely eliminates task interference |
| **Routing Overhead Latency** | N/A | **< 0.02 seconds** | Negligible classification time |

## Applied Science Trade-offs & Ablations

* **Hard Routing vs. Soft Routing:** Hard routing ($\text{argmax}$) executes discrete pointer switching via `model.set_adapter()` with $O(1)$ zero-overhead latency, avoiding the $N$-pass forward overhead and cross-domain weight pollution inherent in soft-routing/weight-blending ($\Delta W_{\text{eff}} = \sum w_i \Delta W_i$).
* **Rank Allocation ($r=32$ for Math):** Expanding the Math adapter rank to $r=32$ provided the parameter capacity necessary for complex multi-pass calculus and modular arithmetic derivations without increasing base memory usage.
* **Native FP16 Execution Precision:** Running the base model in Native FP16 bypassed quantization version conflicts while eliminating dynamic dequantization latency, maintaining maximum token throughput on 16GB GPU hardware.

## Training Pipeline

Each notebook follows the same audited workflow:

1. Install dependencies (`transformers`, `peft`, `accelerate`, `datasets`, `sentence-transformers`)
2. Load base model & tokenizer in Native FP16
3. Token length distribution audit (1024-token context check)
4. Visual inspection of formatted examples
5. Verify response-only loss masking (`labels = -100` on prompts)
6. Preprocess & filter dataset subsets
7. Train and persist LoRA adapters

## Setup

```bash
pip install -r requirements.txt

```

## Running the RouterNet Engine

```python
import torch
from src.engine import RouterNetEngine3Domain

# Initialize 3-Domain Hot-Swapping Engine
engine = RouterNetEngine3Domain(
    base_model_id="Qwen/Qwen3-4B-Instruct-2507",
    math_path="adapters/math",
    creative_path="adapters/creative",
    code_path="adapters/code",
    clf_path="src/router_classifier.joblib"
)

# Automated classification -> Zero-latency Adapter Switch -> Inference
query = "Evaluate the indefinite integral: \int x^2 \cdot e^{3x} \, dx"
result = engine.generate(query, max_tokens=512)

print(f"Routed Domain : [{result['domain']}] ({result['confidence']*100:.1f}% confidence)")
print(f"Response Output:\n{result['response']}")

```

## Milestones

* [x] Train domain-isolated LoRA adapters (`math`, `code`, `creative`)
* [x] Semantic router / classifier to dispatch prompts to domain adapters
* [x] Unified zero-latency multi-adapter inference engine (`src/engine.py`)
* [x] Evaluation benchmarks and comparative baseline analysis

```

```
