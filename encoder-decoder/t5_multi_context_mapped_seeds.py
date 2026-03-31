# -*- coding: utf-8 -*-
"""Fine-tune google/mt5-large on codiesp + mapped corpora with context.

Accepts a --seed argument so that several independent runs can be launched
(one per seed) for reproducibility / variance estimation.

Usage:
    python t5_multi_finetune_context_mapped_seeds.py --seed 42
    python t5_multi_finetune_context_mapped_seeds.py --seed 123
"""

from datetime import datetime
import argparse
import transformers
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)

import os
import sys
import pandas as pd
import numpy as np
import torch

import wandb

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Fine-tune mt5-large with a specific random seed.")
parser.add_argument('--seed', type=int, required=True,
                    help='Random seed for this run')
args_cli = parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────
def calculate_max_tokens(texts, tokenizer):
    token_lengths = [len(tokenizer.encode(text)) for text in texts]
    max_tokens = max(token_lengths)
    if max_tokens % 2 != 0:
        max_tokens += 1
    return max_tokens + 2


def make_prompt(term, sentence):
    return f"Genera una definición para el término: {term} - en la frase: {sentence}"


# ── DATA ─────────────────────────────────────────────────────────────────────
corpus = 'codiesp'
data_path = os.path.join('corpus', corpus, 'sentences_words')
data_train = os.path.join(data_path, 'train_sentences_words_icd10.tsv')
data_dev = os.path.join(data_path, 'dev_sentences_words_icd10.tsv')

train_df = pd.read_csv(data_train, sep='\t', dtype=str)
train_df = train_df.sample(frac=1, random_state=42)
train_df.reset_index(drop=True, inplace=True)

train_df['prompt'] = train_df.apply(
    lambda x: make_prompt(x['words'], x['sentence']), axis=1)

source_column = 'prompt'
target_column = 'descripcion'

train_df = train_df.rename(columns={source_column: "ctext", target_column: "text"})
train_df = train_df[['ctext', 'text']]

eval_df = pd.read_csv(data_dev, sep='\t', dtype=str)
eval_df.reset_index(drop=True, inplace=True)
eval_df['prompt'] = eval_df.apply(
    lambda x: make_prompt(x['words'], x['sentence']), axis=1)

eval_df = eval_df.rename(columns={source_column: "ctext", target_column: "text"})
eval_df = eval_df[['ctext', 'text']]

# ── Mapped corpora ───────────────────────────────────────────────────────────
corpus_mapped = 'mapped_corpora'
mapped_folder = os.path.join('corpus', corpus_mapped)

df_mantra = pd.read_csv(f'{mapped_folder}/Spanish_mantra_icd10.tsv', sep='\t', dtype=str)
df_e3c = pd.read_csv(f'{mapped_folder}/Spanish_e3c_icd10_sentence_layer1.tsv', sep='\t', dtype=str)
df_medterm = pd.read_csv(f'{mapped_folder}/medterm_all_icd10_selected.tsv', sep='\t', dtype=str)

df_mantra['prompt'] = df_mantra.apply(lambda x: make_prompt(x['entity'], x['sentence']), axis=1)
df_mantra = df_mantra[['file_id', 'entity', 'CODE', 'sentence', 'STR', 'prompt']].reset_index(drop=True)

df_e3c['prompt'] = df_e3c.apply(lambda x: make_prompt(x['entity'], x['sentence']), axis=1)
df_e3c = df_e3c[['file_id', 'entity', 'CODE', 'sentence', 'STR', 'prompt']].reset_index(drop=True)

df_medterm['prompt'] = df_medterm.apply(lambda x: make_prompt(x['entity'], x['sentence']), axis=1)
df_medterm = df_medterm[['file_id', 'entity', 'CODE', 'sentence', 'STR', 'prompt']].reset_index(drop=True)

df_all_mapped = pd.concat([df_mantra, df_e3c, df_medterm]).reset_index(drop=True)
df_all_mapped = df_all_mapped.rename(columns={source_column: 'ctext', 'STR': 'text'})
df_all_mapped = df_all_mapped[['ctext', 'text']]
df_all_mapped.to_csv('mapped.tsv', sep='\t', index=False)

print('Total mapped', len(df_all_mapped))
print(df_all_mapped.head())

# ── Combine & clean ──────────────────────────────────────────────────────────
df = pd.concat([train_df, df_all_mapped, eval_df], ignore_index=True)
print('Total train + mapped + eval', len(df))
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)

# Report nulls before filtering
print('Empty ctext', df['ctext'].isnull().sum())
print('Empty text', df['text'].isnull().sum())

# Filter nulls BEFORE splitting so both splits are consistently clean
df = df[df['ctext'].notnull() & df['text'].notnull()]
df.reset_index(drop=True, inplace=True)
df.to_csv('mapped_all_data.tsv', sep='\t', index=False)
print(df.head())

n = len(df) // 10
df_eval = df.iloc[:n].copy()
df_train = df.iloc[n:].copy()
df_train.reset_index(drop=True, inplace=True)
df_eval.reset_index(drop=True, inplace=True)

print('Train', len(df_train))
print('Eval', len(df_eval))
print('Source:', df_train['ctext'].iloc[0])
print('Target', df_train['text'].iloc[0])

tds = Dataset.from_pandas(df_train)
vds = Dataset.from_pandas(df_eval)

medium_datasets = DatasetDict()
medium_datasets['train'] = tds
medium_datasets['validation'] = vds
print(medium_datasets)

# ── Model & tokenizer ───────────────────────────────────────────────────────
model_checkpoint = 'google/mt5-large'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

ctexts = df_train.ctext.to_list()
texts = df_train.text.to_list()
encoder_max_length = calculate_max_tokens(ctexts, tokenizer=tokenizer)
decoder_max_length = calculate_max_tokens(texts, tokenizer=tokenizer)
print('Max encoder length: ', encoder_max_length)
print('Max decoder length: ', decoder_max_length)

# ── Hyperparameters ──────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
TRAIN_EPOCHS = 100
LEARNING_RATE = 1e-5
MAX_SOURCE_LEN = encoder_max_length
MAX_TARGET_LEN = decoder_max_length
N_STEPS = 100

# ── Tokenization ─────────────────────────────────────────────────────────────
def tokenize_data(examples):
    inputs = [text for text in examples["ctext"]]
    model_inputs = tokenizer(inputs,
                             max_length=MAX_SOURCE_LEN,
                             truncation=True)

    labels = tokenizer(text_target=examples["text"],
                       max_length=MAX_TARGET_LEN,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = medium_datasets.map(tokenize_data, batched=True)
print(tokenized_datasets)

data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = evaluate.load("rouge", trust_remote_code=True)


def generated_accuracy(preds, labels):
    accs = []
    for p, l in zip(preds, labels):
        accs.append(1 if p == l else 0)
    return np.mean(accs) * 100


# ── Training run ─────────────────────────────────────────────────────────────
model_name = model_checkpoint.split('/')[-1]
seed = args_cli.seed

print(f'\n{"="*60}')
print(f'Starting training run with seed={seed} on GPU {os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}')
print(f'{"="*60}\n')

# Set random seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
transformers.set_seed(seed)

model_dir = f"output_{corpus}/{model_name}-term-sentence-mapped-seed{seed}"
os.makedirs(model_dir, exist_ok=True)
print('Saving to ', model_dir)

# Per-run WandB init
run = wandb.init(
    project=f"{corpus} MT5",
    name=f"{model_name}-mapped-seed{seed}",
    reinit=True,
    config={
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "VALID_BATCH_SIZE": VALID_BATCH_SIZE,
        "TRAIN_EPOCHS": TRAIN_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "SEED": seed,
        "MAX_SOURCE_LEN": MAX_SOURCE_LEN,
        "MAX_TARGET_LEN": MAX_TARGET_LEN,
        "N_STEPS": N_STEPS,
    }
)

args = Seq2SeqTrainingArguments(
    model_dir,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=N_STEPS,
    save_strategy="epoch",
    do_train=True,
    do_predict=True,
    do_eval=True,
    predict_with_generate=True,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    num_train_epochs=TRAIN_EPOCHS,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=5,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    seed=seed,
    report_to="wandb"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    current_time = datetime.now()
    with open(f'{model_dir}/train_predictions_{current_time}.txt', 'w') as f:
        for item in decoded_preds:
            f.write("%s\n" % item)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value * 100 for key, value in result.items()}
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                       for pred in predictions]
    accuracy = generated_accuracy(decoded_preds, decoded_labels)
    result["Accuracy"] = accuracy
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


wandb.watch(model_init(), log="all")

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print(f'Start training (seed={seed})')
trainer.train()
trainer.save_model()

run.finish()
print(f'Run with seed={seed} finished. Model saved to {model_dir}')
