# -*- coding: utf-8 -*-
"""Evaluate fine-tuned Medical-mT5 models trained with multiple seeds.

Usage examples:
    # Evaluate seeds 42, 123, 456 on test set
    python t5_antidote_eval_seeds.py --seeds 42 123 456 --part test

    # Evaluate a single seed on dev
    python t5_antidote_eval_seeds.py --seeds 42 --part dev

    # Use a specific checkpoint subdirectory instead of the root (best model)
    python t5_antidote_eval_seeds.py --seeds 42 --part test --checkpoint 262517
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from eval.evaluation_utils import calculate_all_scores
from eval.transformer_utils import generate_in_batches

from dotenv import load_dotenv
load_dotenv()


print("CUDA available:", torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate Medical-mT5 models trained with different seeds.")
parser.add_argument('--seeds', type=int, nargs='+', required=True,
                    help='One or more random seeds to evaluate (e.g. --seeds 42 123 456)')
parser.add_argument('--part', type=str, default='test', choices=['test', 'dev'],
                    help='Split to evaluate on (default: test)')
parser.add_argument('--corpus', type=str, default='codiesp',
                    help='Corpus name (default: codiesp)')
parser.add_argument('--checkpoint', type=int, default=None,
                    help='Checkpoint step number to load. If omitted, loads the best model '
                         'saved at the model root (load_best_model_at_end=True).')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Generation batch size (default: 64)')
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
corpus = args.corpus
data_path = os.path.join('corpus', corpus, 'sentences_words')
data_train = os.path.join(data_path, 'train_sentences_words_icd10.tsv')
data_dev   = os.path.join(data_path, 'dev_sentences_words_icd10.tsv')
data_test  = os.path.join(data_path, 'test_sentences_words_icd10.tsv')

corpus_mapped = 'mapped_corpora'
mapped_folder = os.path.join('corpus', corpus_mapped)

# ── Prompt helpers ────────────────────────────────────────────────────────────
def make_prompt(term, sentence):
    return f"Genera una definición para el término: {term} - en la frase: {sentence}"

def make_prefix(term):
    return f"Genera una definición para el término: {term}"

# ── Load & prepare training data (needed for seen/unseen split) ───────────────
print("Loading training data for seen/unseen vocabulary tracking …")

train_df = pd.read_csv(data_train, sep='\t', dtype=str)
train_df = train_df.sample(frac=1, random_state=42)
train_df.reset_index(drop=True, inplace=True)
train_df['prompt'] = train_df.apply(lambda x: make_prompt(x['words'], x['sentence']), axis=1)
train_df = train_df.rename(columns={'prompt': 'ctext', 'descripcion': 'text'})
train_df = train_df[['ctext', 'text', 'words']]

eval_df = pd.read_csv(data_dev, sep='\t', dtype=str)
eval_df.reset_index(drop=True, inplace=True)
eval_df['prompt'] = eval_df.apply(lambda x: make_prompt(x['words'], x['sentence']), axis=1)
eval_df = eval_df.rename(columns={'prompt': 'ctext', 'descripcion': 'text'})
eval_df = eval_df[['ctext', 'text', 'words']]

df_mantra  = pd.read_csv(f'{mapped_folder}/Spanish_mantra_icd10.tsv', sep='\t', dtype=str)
df_e3c     = pd.read_csv(f'{mapped_folder}/Spanish_e3c_icd10_sentence_layer1.tsv', sep='\t', dtype=str)
df_medterm = pd.read_csv(f'{mapped_folder}/medterm_all_icd10_selected.tsv', sep='\t', dtype=str)

for df_m, name_col in [(df_mantra, 'entity'), (df_e3c, 'entity'), (df_medterm, 'entity')]:
    df_m['prompt'] = df_m.apply(lambda x: make_prompt(x[name_col], x['sentence']), axis=1)

df_mantra  = df_mantra[['entity', 'CODE', 'sentence', 'STR', 'prompt']].reset_index(drop=True)
df_e3c     = df_e3c[['entity', 'CODE', 'sentence', 'STR', 'prompt']].reset_index(drop=True)
df_medterm = df_medterm[['entity', 'CODE', 'sentence', 'STR', 'prompt']].reset_index(drop=True)

df_all_mapped = pd.concat([df_mantra, df_e3c, df_medterm]).reset_index(drop=True)
df_all_mapped = df_all_mapped.rename(columns={'prompt': 'ctext', 'STR': 'text', 'entity': 'words'})
df_all_mapped = df_all_mapped[['ctext', 'text', 'words']]

# Full combined set — used only to determine seen/unseen vocabulary
df_all = pd.concat([train_df, eval_df, df_all_mapped], ignore_index=True)
df_all = df_all[df_all['ctext'].notnull() & df_all['text'].notnull()]
print(f"Total training vocabulary size: {df_all['words'].nunique()} unique terms")

# ── Load evaluation split ─────────────────────────────────────────────────────
print(f"\nLoading {args.part.upper()} split …")
if args.part == 'test':
    df_base = pd.read_csv(data_test, sep='\t', dtype=str)
else:
    # Use the 10 % dev split derived the same way as during training
    df_combined = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_combined) // 10
    df_base = df_combined.iloc[:n].copy().reset_index(drop=True)

df_base['prompt'] = df_base.apply(lambda x: make_prompt(x['words'], x['sentence']), axis=1)
df_base['prefix'] = df_base.apply(lambda x: make_prefix(x['words']), axis=1)
df_base.reset_index(drop=True, inplace=True)
if 'id' in df_base.columns:
    df_base['id'] = df_base['id'].astype(str)

print(f"{args.part.upper()} set size: {len(df_base)}")

source_column = 'prompt'

# ── Evaluate each seed ────────────────────────────────────────────────────────
summary_rows = []

for seed in args.seeds:
    print(f"\n{'='*60}")
    print(f"Evaluating seed={seed}")
    print(f"{'='*60}")

    model_name = 'Medical-mT5-large'
    model_dir = os.path.join(f'output_{corpus}', f'{model_name}-term-sentence-mapped-v2-seed{seed}')

    if args.checkpoint is not None:
        eval_model_dir = os.path.join(model_dir, f'checkpoint-{args.checkpoint}')
    else:
        # load_best_model_at_end saves best model to the root of model_dir
        eval_model_dir = model_dir

    if not os.path.isdir(eval_model_dir):
        print(f"[WARNING] Model directory not found: {eval_model_dir} — skipping.")
        continue

    pred_path   = os.path.join(eval_model_dir, f'{args.part}_predictions.tsv')
    scores_path = os.path.join(eval_model_dir, f'{args.part}_scores.tsv')

    if os.path.isfile(pred_path):
        print(f"Predictions file already exists, loading from: {pred_path}")
        df_out = pd.read_csv(pred_path, sep='\t', dtype=str)
        predictions = df_out['pred'].to_list()
        descriptions = df_out['descripcion'].to_list()
        print(f"Loaded {len(predictions)} predictions — skipping generation.")
    else:
        print(f"Loading model from: {eval_model_dir}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(eval_model_dir, cache_dir=cache_folder)
        except (ValueError, OSError):
            base_checkpoint = 'HiTZ/Medical-mT5-large'
            print(f"[WARNING] Tokenizer not found in {eval_model_dir}, "
                  f"falling back to base model tokenizer: {base_checkpoint}")
            tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, cache_dir=cache_folder)
        model = MT5ForConditionalGeneration.from_pretrained(
            eval_model_dir, cache_dir=cache_folder
        ).to(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        generation_config = model.generation_config
        print(generation_config)

        # ── Generate predictions ──────────────────────────────────────────
        predictions = generate_in_batches(
            df_base[source_column].to_list(),
            args.batch_size,
            tokenizer=tokenizer,
            model=model,
            generation_config=generation_config,
        )
        print(f"Predictions generated: {len(predictions)}")

        descriptions = df_base['descripcion'].to_list()

        # ── Overall scores ────────────────────────────────────────────────
        df_scores, df_list_scores = calculate_all_scores(predictions, descriptions)
        df_scores.to_csv(scores_path, sep='\t', index=False)
        print(f"PERFORMANCE (seed={seed})")
        print(df_scores.to_string(index=False))

        # Save predictions
        df_out = df_base.copy()
        df_out['pred'] = predictions
        df_out['semscore'] = df_list_scores['SEMSCORE'].to_list()
        df_out['bertscore'] = df_list_scores['BERTSCORE'].to_list()
        df_out.to_csv(pred_path, sep='\t', index=False)
        print(f"Predictions saved to: {pred_path}")

        # Free GPU memory before next seed
        del model
        torch.cuda.empty_cache()

    # ── Scores (recompute from loaded predictions if file was cached) ─────
    if not os.path.isfile(scores_path):
        df_scores, _ = calculate_all_scores(predictions, descriptions)
        df_scores.to_csv(scores_path, sep='\t', index=False)
    else:
        df_scores = pd.read_csv(scores_path, sep='\t')
    print(f"PERFORMANCE (seed={seed})")
    print(df_scores.to_string(index=False))

    # ── Unseen / seen split ───────────────────────────────────────────────
    known_words = set(df_all['words'].dropna().values)
    df_unseen = df_out[~df_out['words'].isin(known_words)]
    df_seen   = df_out[ df_out['words'].isin(known_words)]
    print(f"Seen: {len(df_seen)}  |  Unseen: {len(df_unseen)}")

    if len(df_unseen) > 0:
        df_scores_unseen, _ = calculate_all_scores(
            df_unseen['pred'].to_list(), df_unseen['descripcion'].to_list()
        )
        unseen_path = os.path.join(eval_model_dir, f'{args.part}_unseen_scores.tsv')
        df_scores_unseen.to_csv(unseen_path, sep='\t', index=False)
        print("UNSEEN PERFORMANCE")
        print(df_scores_unseen.to_string(index=False))

    if len(df_seen) > 0:
        df_scores_seen, _ = calculate_all_scores(
            df_seen['pred'].to_list(), df_seen['descripcion'].to_list()
        )
        seen_path = os.path.join(eval_model_dir, f'{args.part}_seen_scores.tsv')
        df_scores_seen.to_csv(seen_path, sep='\t', index=False)
        print("SEEN PERFORMANCE")
        print(df_scores_seen.to_string(index=False))

    # Collect one-row summary for end-of-script overview
    row = {'seed': seed}
    for col in df_scores.columns:
        row[col] = df_scores[col].iloc[0]
    summary_rows.append(row)

# ── Cross-seed summary ────────────────────────────────────────────────────────
if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    summary_path = f'models_{args.part}_scores_seeds.tsv'
    df_summary.to_csv(summary_path, sep='\t', index=False)
    print(f"\n{'='*60}")
    print("CROSS-SEED SUMMARY")
    print(df_summary.to_string(index=False))

    numeric_cols = df_summary.select_dtypes(include='number').drop(columns=['seed'], errors='ignore').columns
    means = df_summary[numeric_cols].mean()
    stds  = df_summary[numeric_cols].std()
    print("\nMean ± Std across seeds:")
    for col in numeric_cols:
        print(f"  {col}: {means[col]:.4f} ± {stds[col]:.4f}")

    print(f"\nSummary saved to: {summary_path}")
