# RouterNet

A domain-specialized LoRA adapter suite for **Qwen3-4B-Instruct** — three lightweight adapters fine-tuned for creative writing, code generation, and mathematical reasoning, designed to serve as routing targets in a mixture-of-experts style setup.

## Repository Structure

```
RouterNet/
├── notebooks/                  # End-to-end adapter training pipelines
│   ├── 01-train-creative-adapter.ipynb
│   ├── 02-train-code-adapter.ipynb
│   └── 03-train-math-adapter.ipynb
├── adapters/                   # Trained PEFT/LoRA checkpoints
│   ├── creative/
│   ├── code/
│   └── math/
├── data/                       # Datasets & preprocessing outputs
├── src/                        # Router / inference code
└── docs/                       # Documentation
```

## Adapters

| Adapter    | Domain              | Datasets | LoRA rank | LR     |
|------------|---------------------|----------|-----------|--------|
| `creative` | Creative writing    | [euclaise/WritingPrompts_curated](https://huggingface.co/datasets/euclaise/WritingPrompts_curated) | 16 | 5e-5 |
| `code`     | Code generation     | [nickrosh/Evol-Instruct-Code-80k-v1](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) | 16 | 5e-5 |
| `math`     | Math reasoning      | [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 32 | 2e-4 |

**Base model:** [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

## Training Pipeline

Each notebook follows the same audited workflow:

1. Install dependencies (`transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`)
2. Load tokenizer & dataset
3. Token length distribution audit (1024-token context check)
4. Visual inspection of formatted examples
5. Verify response-only loss masking (`labels = -100` on prompts)
6. Preprocess & filter
7. Train the LoRA adapter

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** Training uses QLoRA-style bitsandbytes quantization — a CUDA GPU is required.

## Loading an Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", device_map="auto")
model = PeftModel.from_pretrained(base, "adapters/math")
tokenizer = AutoTokenizer.from_pretrained("adapters/math")
```

## Roadmap

- [ ] Router / classifier to dispatch prompts to domain adapters
- [ ] Unified multi-adapter inference server (`src/`)
- [ ] Evaluation benchmarks per domain
