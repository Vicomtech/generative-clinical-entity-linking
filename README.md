# Medical Entity Linking to ICD-10: Baselines and Generative Approaches

This repository contains the code and data for reproducing the experiments described in the paper:

> **[Paper Title Here]**  
> [Author(s)]  
> [Venue / Journal, Year]

## Overview

We address the task of **clinical entity linking** — mapping medical terms in clinical Spanish text to **ICD-10 codes** — by generating natural language definitions that are then linked to the ICD-10 terminology. We evaluate and compare:

1. **Baselines**: String matching, bi-encoder (SapBERT + FAISS), and cross-encoder re-ranking
2. **Encoder-decoder models**: mT5, Medical-mT5, and mBART fine-tuned for definition generation
3. **Large Language Models**: Meta-Llama-3.1-8B fine-tuned with LoRA (4-bit quantization)

All approaches are evaluated on the [CodiEsp](https://temu.bsc.es/codiesp/) shared task corpus.

## Repository Structure

```
.
├── .env                          # Environment variables (copy from .env.example)
├── requirements.txt
├── evaluate_hf.py                # ← Evaluate the published HF model on the test set
├── hf_example.py                 # Minimal single-example inference script
├── corpus/                       # CodiEsp corpus + mapped corpora
│   ├── codiesp/                  # Train/dev/test splits with sentence-level annotations
│   ├── cross_encoder_corpus_codiesp/
│   ├── mapped_corpora/           # Spanish Mantra, E3C, MedTerm mapped to ICD-10
│   └── ...
├── icd_10_es/                    # ICD-10 ontology files (Spanish, 2020)
├── baselines/
│   ├── string_match/             # Exact string matching baseline
│   ├── bi-encoder/               # SapBERT + FAISS retrieval
│   └── cross-encoder/            # Cross-encoder re-ranking
├── encoder-decoder/              # mT5 and mBART fine-tuning & evaluation
├── decoder-only/                 # Llama 3.1-8B fine-tuning & evaluation
├── eval/                         # Shared evaluation utilities
├── utils/                        # Shared transformer/FAISS utilities
├── analysis/                     # Performance aggregation scripts
└── link_generated/               # Semantic similarity linking for generated outputs
```

## Best Model — Quick Evaluation

The best model (Medical-mT5-large fine-tuned on CodiEsp + mapped corpora) is published on HuggingFace:

**[ezotova/medical-mt5-clinical-el-spanish](https://huggingface.co/ezotova/medical-mt5-clinical-el-spanish)**

### Run full test-set evaluation

`evaluate_hf.py` downloads the model automatically and evaluates it on the CodiEsp test set. No training artefacts are required.

```bash
# with default settings (batch_size=8, output saved to hf_eval/)
python evaluate_hf.py

# all options
python evaluate_hf.py \
    --model    ezotova/medical-mt5-clinical-el-spanish \
    --corpus   codiesp \
    --batch_size 8 \
    --output_dir hf_eval
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `ezotova/medical-mt5-clinical-el-spanish` | HuggingFace model ID or local path |
| `--corpus` | `codiesp` | Corpus name (folder inside `corpus/`) |
| `--batch_size` | `8` | Generation batch size (lower if OOM on GPU) |
| `--output_dir` | `hf_eval` | Directory for predictions and score files |

Output files written to `--output_dir`:

- `<model>_test_predictions.tsv` — test rows with `pred`, `semscore`, `bertscore` columns
- `<model>_test_scores.tsv` — aggregate ROUGE / BLEU / METEOR / BERTScore / SemScore

If predictions already exist the script skips generation and recomputes scores only.

### Single-example inference

```python
from transformers import AutoConfig, AutoTokenizer, MT5ForConditionalGeneration

model_id = "ezotova/medical-mt5-clinical-el-spanish"

config    = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = MT5ForConditionalGeneration.from_pretrained(model_id, config=config)

def make_prompt(term, sentence):
    return f"Genera una definición para el término: {term} - en la frase: {sentence}"

term     = "inestabilidad a la marcha"
sentence = "El paciente refiere sensación de inestabilidad a la marcha y temblores en las manos."

inputs  = tokenizer(make_prompt(term, sentence), return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64, num_beams=5, early_stopping=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **Note:** `AutoConfig` must be loaded first and passed to `from_pretrained`. The model was
> fine-tuned with independent `lm_head` weights, so passing `config` prevents the default
> `tie_word_embeddings` behaviour from overwriting them.
## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested with NVIDIA A100 / V100)
- [HuggingFace account](https://huggingface.co/) with access to gated models (Llama 3.1)

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for BLEU/METEOR evaluation)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### Configuration

1. **HuggingFace token**: Required for downloading gated models (e.g., Llama 3.1).

   Edit `.env` at the repo root and fill in `HF_TOKEN`:

   ```bash
   HF_TOKEN=hf_your_token_here
   ```

2. **Cache location**: Set `CACHE_FOLDER` in `.env` to your HuggingFace model cache directory.

3. **GPU selection**: `CUDA_VISIBLE_DEVICES` is also controlled via `.env`. Adjust to match your hardware.

## Data

### Included in this repository

## Reproducing Experiments

### 1. String Matching Baseline

```bash
cd baselines/string_match
python string_match.py
```

### 2. Bi-Encoder + Cross-Encoder Baseline

```bash
cd baselines/bi-encoder
# Step 1: Encode and retrieve with SapBERT + FAISS
python 01_model_sent_codiesp.py

# Step 2: Evaluate bi-encoder
python 02_calc_performance.py

# Step 3: Re-rank with cross-encoder
cd ../cross-encoder
python 03_cross_encoder_codiesp.py
```

### 3. Encoder-Decoder (mT5)

```bash
cd encoder-decoder

# Fine-tune Medical-mT5 with context and mapped corpora (runs 3 seeds in parallel)
python launch_seeds.py

# Evaluate
python medical_t5_eval_seeds.py --seeds 42 123 456 --corpus codiesp --part test
```

### 4. Encoder-Decoder (mBART)

```bash
cd encoder-decoder

# Fine-tune
python mbart.py

# Evaluate
python mbart_eval.py
```

### 5. LLM (Llama 3.1-8B)

```bash
cd decoder-only

# Prepare training corpus
python corpus_llama_context.py

# Fine-tune with LoRA
python finetune_llama.py

# Evaluate
python eval_llama.py
```

### 6. Analysis

```bash
cd analysis

# Calculate performance metrics
python calc_performance.py

# Aggregate across seeds
python calc_mean.py
```


## Training Configuration

## Evaluation Metrics

- **Entity Linking**: Accuracy, Precision, Recall, F1, Recall@K (K ∈ {1, 8, 16, 32, 64, 128})
- **Definition Generation**: ROUGE-L, BLEU, METEOR, BERTScore
- **Semantic Similarity**: SemScore (cosine similarity with all-mpnet-base-v2)
- **Seen vs. Unseen**: Separate evaluation on terms seen/unseen during training

## Citation

If you use this code or data, please cite:

```bibtex
@article{zotova2026generative,
  title={Generative Models for Clinical Entity Linking in Spanish},
  author={Zotova, Elena and Cuadros, Montse and Rigau, German},
  journal={Preprint submitted to Elsevier: Available at SSRN 5976036}, 
  year={2026},
  url={https://github.com/Vicomtech/generative-clinical-entity-linking}
}
```

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC 4.0).

You are free to share and adapt the material for non-commercial purposes, provided appropriate credit is given. See the [LICENSE](LICENSE) file for the full terms.

## Acknowledgments

<!-- Add funding, institutions, shared task organizers, etc. -->
- CodiEsp shared task organizers (BSC)
- MedProcNER shared task organizers (BSC)
- LivingNER shared task organizers (BSC)
