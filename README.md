Algorithmic Code AnalystOverviewThe Algorithmic Code Analyst is a Parameter-Efficient Fine-Tuned (PEFT) Large Language Model designed to bridge the gap between deep learning and core computer science. Fine-tuned on the 3-Billion parameter StarCoder2 base architecture, this model ingests raw C++ functions and outputs a structured, multi-objective XML analysis containing a professional docstring, Time Complexity, and Space Complexity.Architecture & Data PipelineTraining a multi-objective code analysis model required bypassing standard dataset limitations via synthetic data engineering and memory optimization.Synthetic Augmentation: Leveraged Hugging Face's XLCoST base dataset and constructed a fault-tolerant Python pipeline using the Gemini API and Llama 3 to synthetically generate precise Big O labels and active-verb docstrings.Data Sanitation: Engineered runtime preprocessing filters to strip raw text artifacts (NEW_LINE) and enforce a strict 100% syntactic validation pass rate before tokenization.Quantization & Training: Deployed a 4-bit NF4 quantized QLoRA training loop via a Google Colab T4 GPU. Optimized VRAM usage via gradient accumulation (effective batch size of 8) to train the LoRA adapters without memory overflow.Model Output SchemaThe model is trained to output a strict, parseable XML format for easy downstream integration:XML<analysis>
<docstring>
Calculates the maximum subarray sum using Kadane's algorithm.
</docstring>
<time_complexity>O(N)</time_complexity>
<space_complexity>O(1)</space_complexity>
</analysis>
Performance & Known Limitations (v1.0)This first iteration was trained on a highly curated subset of 300 synthetic samples to test structural style transfer capabilities.Structural Success: The QLoRA adapters successfully transferred the structural style. The model boasts a near 100% adherence rate to the custom XML schema and accurately identifies basic loops and pointers to predict constant or linear complexities.Logical Limitations (Overfitting on Syntax): Because the v1.0 training dataset was restricted to 300 samples, the model exhibits pattern-matching hallucinations on unseen algorithmic paradigms. For example, when evaluating C++ code containing std::sort, the 3B model defaults to predicting $O(N)$ based on surrounding for loops, failing to logically deduce the $O(N \log N)$ complexity of the underlying Introsort.Next Steps & Future IterationsDataset Scaling: Scale the synthetic generation pipeline to 5,000+ highly diverse examples (incorporating dynamic programming, trees, and graphs) to force the model away from syntax pattern-matching and toward deeper algorithmic reasoning.LLM-as-a-Judge Evaluation: Implement an automated evaluation pipeline where a larger teacher model (e.g., GPT-4o) scores the semantic accuracy of the predicted complexities, moving away from rigid BLEU score metrics.Quick Start (Inference)To run inference using the fine-tuned adapters:Pythonimport torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "bigcode/starcoder2-3b"
adapter_dir = "path/to/downloaded/adapters"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto"
)

# Inject custom adapters
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

# Run inference...
