# Medical Entity Linking to ICD-10: Baselines and Generative Approaches

This repository contains the code and data for reproducing the experiments described in the paper:

> **Generative Models for Clinical Entity Linking in Spanish**  
> Elena Zotova, Montse Cuadros and German Rigau  
> Array, Elsevier, 2026

## Overview

We address the task of **clinical entity linking** — mapping medical terms in clinical Spanish text to **ICD-10 codes** — by generating natural language definitions that are then linked to the ICD-10 terminology. We evaluate and compare:

1. **Baselines**: String matching, bi-encoder (SapBERT + FAISS), and cross-encoder re-ranking
2. **Encoder-decoder models**: mT5, Medical-mT5, and mBART fine-tuned for definition generation
3. **Large Language Models**: Meta-Llama-3.1-8B fine-tuned with LoRA (4-bit quantization)

All approaches are evaluated on the [CodiEsp](https://temu.bsc.es/codiesp/) shared task corpus.

## Repository Structure

| Path | Description |
|------|-------------|
| `.env` | Environment variables (copy from `.env.example`) |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`evaluate_hf.py`](evaluate_hf.py) | Evaluate the published HF model on the test set |
| [`hf_example.py`](hf_example.py) | Minimal single-example inference script |
| [`corpus/`](corpus/) | CodiEsp corpus + mapped corpora |
| &nbsp;&nbsp;[`codiesp/`](corpus/codiesp/) | Train/dev/test splits with sentence-level annotations |
| &nbsp;&nbsp;[`cross_encoder_corpus_codiesp/`](corpus/cross_encoder_corpus_codiesp/) | Cross-encoder training/dev corpus |
| &nbsp;&nbsp;[`mapped_corpora/`](corpus/mapped_corpora/) | Spanish Mantra, E3C, MedTerm mapped to ICD-10 |
| [`icd_10_es/`](icd_10_es/) | ICD-10 ontology files (Spanish, 2020) |
| [`baselines/`](baselines/) | Baseline experiments |
| &nbsp;&nbsp;[`string_match/`](baselines/string_match/) | Exact string matching baseline |
| &nbsp;&nbsp;[`bi-encoder/`](baselines/bi-encoder/) | SapBERT + FAISS retrieval |
| &nbsp;&nbsp;[`cross-encoder/`](baselines/cross-encoder/) | Cross-encoder re-ranking |
| [`encoder-decoder/`](encoder-decoder/) | mT5 and mBART fine-tuning & evaluation |
| [`decoder-only/`](decoder-only/) | Llama 3.1-8B fine-tuning & evaluation |
| [`eval/`](eval/) | Shared evaluation utilities |
| [`utils/`](utils/) | Shared transformer/FAISS utilities |
| [`analysis/`](analysis/) | Performance aggregation scripts |
| [`link_generated/`](link_generated/) | Semantic similarity linking for generated outputs |

## Best Model — Quick Evaluation

The best model (Medical-mT5-large fine-tuned on CodiEsp + mapped corpora) is published on HuggingFace:

**[ezotova/medical-mt5-clinical-el-spanish](https://huggingface.co/ezotova/medical-mt5-clinical-el-spanish)**


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

## Training Data

The model was fine-tuned on three Spanish clinical NLP datasets:

| Dataset | Description |
|---|---|
| [Biomedical-TeMU/CodiEsp_corpus](https://huggingface.co/datasets/Biomedical-TeMU/CodiEsp_corpus) | CodiEsp shared task corpus — Spanish clinical cases annotated with ICD-10 codes (diagnoses and procedures) |
| [DrBenchmark/E3C](https://huggingface.co/datasets/DrBenchmark/E3C) | European clinical corpus with entity annotations across multiple languages, including Spanish |
| [IIC/CT-EBM-SP](https://huggingface.co/datasets/IIC/CT-EBM-SP) | Spanish clinical trial corpus annotated for biomedical entities |
| [Mantra GSC](https://huggingface.co/datasets/bigbio/mantra_gsc) | Medline abstract titles, drug labels, biomedical patent claims in Spanish |


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

