"""
evaluate_hf.py
Evaluate the Medical-mT5 model published on the HuggingFace Hub against the
CodiEsp test set.  No local training artefacts are needed — the model is
downloaded on first run and cached by HuggingFace.

Quick start
-----------
    pip install -r requirements.txt   # once
    python evaluate_hf.py             # defaults: codiesp corpus

All options
-----------
    python evaluate_hf.py --corpus codiesp --batch_size 64 \
                           --model ezotova/medical-mt5-clinical-el-spanish \
                           --output_dir hf_eval
"""

import argparse
import os

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, MT5ForConditionalGeneration

from eval.evaluation_utils import calculate_all_scores
from eval.transformer_utils import generate_in_batches

from dotenv import load_dotenv
load_dotenv()

print("CUDA available:", torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Evaluate Medical-mT5 downloaded from HuggingFace Hub on the test set."
)
parser.add_argument(
    "--model",
    type=str,
    default="ezotova/medical-mt5-clinical-el-spanish",
    help="HuggingFace model ID or local path (default: ezotova/medical-mt5-clinical-el-spanish)",
)
parser.add_argument(
    "--corpus",
    type=str,
    default="codiesp",
    help="Corpus name (default: codiesp)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Generation batch size (default: 8)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="hf_eval",
    help="Directory where predictions and scores are saved (default: hf_eval)",
)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
cache_folder = (
    os.environ.get("CACHE_FOLDER") or os.environ.get("HF_HOME") or None
)

corpus = args.corpus
data_path = os.path.join("corpus", corpus)
data_test = os.path.join(data_path, "test_sentences_words_icd10.tsv")

os.makedirs(args.output_dir, exist_ok=True)

# ── Prompt helper ─────────────────────────────────────────────────────────────
def make_prompt(term, sentence):
    return f"Genera una definición para el término: {term} - en la frase: {sentence}"

# ── Load test set ─────────────────────────────────────────────────────────────
print("Loading TEST split …")
df_base = pd.read_csv(data_test, sep="\t", dtype=str)
df_base["prompt"] = df_base.apply(
    lambda x: make_prompt(x["words"], x["sentence"]), axis=1
)
df_base.reset_index(drop=True, inplace=True)
if "id" in df_base.columns:
    df_base["id"] = df_base["id"].astype(str)

print(f"TEST set size: {len(df_base)}")

# ── Output file paths ─────────────────────────────────────────────────────────
safe_model_name = args.model.replace("/", "_")
pred_path   = os.path.join(args.output_dir, f"{safe_model_name}_test_predictions.tsv")
scores_path = os.path.join(args.output_dir, f"{safe_model_name}_test_scores.tsv")

# ── Load or generate predictions ──────────────────────────────────────────────
if os.path.isfile(pred_path):
    print(f"Predictions file already exists, loading from: {pred_path}")
    df_out = pd.read_csv(pred_path, sep="\t", dtype=str)
    predictions  = df_out["pred"].tolist()
    descriptions = df_out["descripcion"].tolist()
    print(f"Loaded {len(predictions)} predictions — skipping generation.")
else:
    # Download (or load from cache) the HuggingFace model.
    # NOTE: The model was fine-tuned with broken weight-tying so lm_head.weight
    # is stored as an independent tensor.  Loading with an explicit config
    # (tie_word_embeddings=True from the Hub) would silently overwrite lm_head;
    # we therefore load the config first and keep it as-is.
    print(f"\nDownloading / loading model: {args.model}")
    config    = AutoConfig.from_pretrained(args.model, cache_dir=cache_folder)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_folder)
    model     = MT5ForConditionalGeneration.from_pretrained(
        args.model, config=config, cache_dir=cache_folder
    ).to(device)

    print(model.generation_config)

    predictions = generate_in_batches(
        df_base["prompt"].tolist(),
        args.batch_size,
        tokenizer=tokenizer,
        model=model,
        generation_config=model.generation_config,
    )
    print(f"Predictions generated: {len(predictions)}")

    descriptions = df_base["descripcion"].tolist()

    # Overall scores
    df_scores, df_list_scores = calculate_all_scores(predictions, descriptions)
    df_scores.to_csv(scores_path, sep="\t", index=False)
    print("\nPERFORMANCE")
    print(df_scores.to_string(index=False))

    # Save predictions with per-sample scores
    df_out = df_base.copy()
    df_out["pred"]      = predictions
    df_out["semscore"]  = df_list_scores["SEMSCORE"].tolist()
    df_out["bertscore"] = df_list_scores["BERTSCORE"].tolist()
    df_out.to_csv(pred_path, sep="\t", index=False)
    print(f"Predictions saved to : {pred_path}")

    del model
    torch.cuda.empty_cache()

# ── Scores (recompute from cached predictions when generation was skipped) ────
if not os.path.isfile(scores_path):
    df_scores, _ = calculate_all_scores(predictions, descriptions)
    df_scores.to_csv(scores_path, sep="\t", index=False)
else:
    df_scores = pd.read_csv(scores_path, sep="\t")

print(f"\nScores saved to      : {scores_path}")
print("\nOVERALL PERFORMANCE")
print(df_scores.to_string(index=False))

