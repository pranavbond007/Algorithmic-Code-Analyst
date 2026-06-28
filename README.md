# 🤖 Algorithmic Code Analyst

> A Parameter-Efficient Fine-Tuned LLM that ingests raw C++ functions and outputs structured Big O complexity analysis with professional docstrings.

<p align="center">
  <img src="https://img.shields.io/badge/Model-StarCoder2--3B-blue?style=for-the-badge&logo=huggingface" />
  <img src="https://img.shields.io/badge/Method-QLoRA%20%2F%20PEFT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Output-XML%20Schema-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-v1.0%20Research-yellow?style=for-the-badge" />
</p>

---

## 📖 Overview

The **Algorithmic Code Analyst** bridges the gap between deep learning and core computer science. Fine-tuned on the **3-Billion parameter StarCoder2** base architecture using **QLoRA (PEFT)**, this model takes raw C++ functions as input and outputs a structured, multi-objective XML analysis containing:

- ✅ A professional **docstring** (active-verb, concise)
- ✅ **Time Complexity** in Big O notation
- ✅ **Space Complexity** in Big O notation

---

## 🏗️ Architecture & Data Pipeline

Training a multi-objective code analysis model required bypassing standard dataset limitations via **synthetic data engineering** and aggressive **memory optimization**.

```
Raw C++ Code  ──►  QLoRA Fine-tuned StarCoder2-3B  ──►  Structured XML Analysis
```

### 1. Synthetic Data Augmentation
- Bootstrapped from Hugging Face's **XLCoST** base dataset
- Built a fault-tolerant Python pipeline using the **Gemini API + Llama 3** to synthetically generate:
  - Precise Big O labels (time & space)
  - Active-verb professional docstrings

### 2. Data Sanitation
- Runtime preprocessing filters to strip raw text artifacts (e.g., `NEW_LINE` tokens)
- Enforced a strict **100% syntactic validation pass rate** before tokenization

### 3. Quantization & Training
| Config | Value |
|---|---|
| Base Model | `bigcode/starcoder2-3b` |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Method | QLoRA (PEFT) |
| Hardware | Google Colab T4 GPU |
| Effective Batch Size | 8 (via gradient accumulation) |
| Training Samples | 300 (v1.0 research subset) |

---

## 📤 Model Output Schema

The model outputs a strict, parseable **XML format** for easy downstream integration:

```xml
<analysis>
  <docstring>
    Calculates the maximum subarray sum using Kadane's algorithm.
  </docstring>
  <time_complexity>O(N)</time_complexity>
  <space_complexity>O(1)</space_complexity>
</analysis>
```

---

## 🚀 Quick Start (Inference)

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes accelerate
```

### Run Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "bigcode/starcoder2-3b"
ADAPTER_DIR   = "path/to/downloaded/adapters"  # Update this path

# ── Load Tokenizer ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# ── Load 4-bit Quantized Base Model ───────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

# ── Inject LoRA Adapters ───────────────────────────────────────────────────────
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# ── Inference ─────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """### Instruction:
Analyze the following C++ function and provide a docstring, time complexity, and space complexity in XML format.

### C++ Code:
{code}

### Analysis:
"""

cpp_function = """
int maxSubarraySum(vector<int>& nums) {
    int maxSum = nums[0], currentSum = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]);
        maxSum = max(maxSum, currentSum);
    }
    return maxSum;
}
"""

inputs = tokenizer(
    PROMPT_TEMPLATE.format(code=cpp_function),
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Expected Output

```xml
<analysis>
  <docstring>
    Finds the contiguous subarray with the largest sum using Kadane's algorithm.
  </docstring>
  <time_complexity>O(N)</time_complexity>
  <space_complexity>O(1)</space_complexity>
</analysis>
```

---

## 📊 Performance & Known Limitations (v1.0)

This first iteration was trained on a curated subset of **300 synthetic samples** to validate structural style transfer capabilities.

### ✅ What Works Well
| Capability | Status |
|---|---|
| XML schema adherence | ~100% |
| Linear complexity detection (`for` loops) | ✅ Reliable |
| Constant space complexity (`O(1)`) | ✅ Reliable |
| Pointer-based traversal patterns | ✅ Reliable |
| Docstring generation (active-verb style) | ✅ Reliable |

### ⚠️ Known Limitations (Overfitting on Syntax)

Due to the 300-sample training scope, the model exhibits **pattern-matching hallucinations** on unseen algorithmic paradigms:

> **Example:** Code containing `std::sort` is predicted as `O(N)` based on surrounding `for` loops — the model fails to deduce the underlying `O(N log N)` Introsort complexity.

| Paradigm | Expected | v1.0 Predicted |
|---|---|---|
| `std::sort` | `O(N log N)` | `O(N)` ⚠️ |
| Nested loops with pruning | `O(N log N)` | `O(N²)` ⚠️ |
| Recursive Fibonacci | `O(2^N)` | `O(N)` ⚠️ |
| DP with memoization | `O(N·M)` | `O(N)` ⚠️ |

---

## 🔮 Roadmap & Future Iterations

### v2.0 — Scale & Reasoning
- [ ] **Dataset Scaling:** Grow synthetic pipeline to **5,000+ diverse examples**, covering:
  - Dynamic programming (DP tables, memoization)
  - Tree traversals (BFS, DFS, recursive)
  - Graph algorithms (Dijkstra, Bellman-Ford)
  - Divide-and-conquer patterns
- [ ] **LLM-as-a-Judge Evaluation:** Automated pipeline where a teacher model (e.g., GPT-4o) scores semantic accuracy of predicted complexities — replacing rigid BLEU metrics
- [ ] **Harder negative sampling:** Inject intentionally tricky examples (e.g., `std::sort` inside loops) to force deeper algorithmic reasoning

### v3.0 — Deployment
- [ ] Hugging Face Spaces demo
- [ ] REST API wrapper for IDE plugin integration
- [ ] Support for Python and Java analysis

---

## 🧩 Project Structure

```
algorithmic-code-analyst/
├── adapters/               # Fine-tuned LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── data/
│   ├── raw/                # XLCoST base dataset samples
│   ├── synthetic/          # Gemini/Llama-generated labels
│   └── processed/          # Tokenized, sanitized training data
├── training/
│   ├── generate_synthetic_data.py
│   ├── train_qlora.py
│   └── validate_schema.py
├── inference/
│   └── run_inference.py
├── notebooks/
│   └── training_colab.ipynb
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | [StarCoder2-3B](https://huggingface.co/bigcode/starcoder2-3b) |
| Fine-Tuning Method | QLoRA via [PEFT](https://github.com/huggingface/peft) |
| Quantization | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) NF4 4-bit |
| Training Framework | [Hugging Face Transformers](https://github.com/huggingface/transformers) + [TRL](https://github.com/huggingface/trl) |
| Synthetic Data | Gemini API + Llama 3 |
| Base Dataset | [XLCoST](https://huggingface.co/datasets/codeparrot/xlcost-text-to-code) |
| Training Hardware | Google Colab (NVIDIA T4 GPU) |

---

## 📄 License

This project is released under the **MIT License**. The base model [StarCoder2](https://huggingface.co/bigcode/starcoder2-3b) is subject to the [BigCode Open RAIL-M v1 License](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement).

---

## 🙏 Acknowledgements

- [BigCode Project](https://www.bigcode-project.org/) for the StarCoder2 base model
- [Hugging Face](https://huggingface.co/) for the PEFT, Transformers, and TRL libraries
- [Tim Dettmers](https://github.com/TimDettmers) for BitsAndBytes quantization

---

<p align="center">
  Built with ❤️ — Bridging Deep Learning & Algorithm Analysis
</p>
