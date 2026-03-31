# -*- coding: utf-8 -*-
import transformers
from datasets import load_dataset, Dataset, DatasetDict
import evaluate
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MBartForConditionalGeneration, GenerationConfig

import os
import sys
import pandas as pd 
import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

import nltk
nltk.download('punkt')
from transformers import AutoTokenizer

# WandB – Import the wandb library
import wandb

def calculate_max_tokens(texts, tokenizer):
   token_lengths = [len(tokenizer.encode(text)) for text in texts]
   max_tokens = max(token_lengths)
   if max_tokens % 2 != 0:
        max_tokens += 1
   return max_tokens + 2


## DATA
corpus = 'codiesp'
data_path = os.path.join('corpus', corpus, 'sentence2description_t5')
data_train = os.path.join(data_path, 'train_sentence2description_t5.tsv')
data_dev = os.path.join(data_path, 'dev_sentence2description_t5.tsv')

train_df = pd.read_csv(data_train, sep='\t')
train_df = train_df.sample(frac=1, random_state=42)
train_df.reset_index(drop=True, inplace=True) # necessary for dataloader

prefix = "Genera una definición para el término entre <label>: "
source_column = 'sentence'
target_column = 'description'

print(train_df.head())
train_df = train_df.rename(columns={source_column: "ctext", target_column: "text"})
train_df = train_df[['ctext', 'text']]
train_df.ctext = prefix + train_df.ctext
print(train_df.head())
print(len(train_df))

eval_df = pd.read_csv(data_dev, sep='\t')
eval_df.reset_index(drop=True, inplace=True)
eval_df = eval_df.rename(columns={source_column: "ctext", target_column: "text"})
eval_df = eval_df[['ctext', 'text']]
eval_df.ctext = prefix + eval_df.ctext
print(eval_df.head())

train_data_txt = Dataset.from_pandas(train_df)
validation_data_txt = Dataset.from_pandas(eval_df)
print(train_data_txt)

"""Data preprocessing"""
wandb.init(project=f"{corpus}_mBART_50")

model_checkpoint = 'facebook/mbart-large-50'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

encoder_max_length = calculate_max_tokens(train_df.ctext.to_list(), tokenizer=tokenizer)
decoder_max_length = calculate_max_tokens(train_df.text.to_list(), tokenizer=tokenizer)
print('Max encoder length: ', encoder_max_length)
print('Max decoder length: ', decoder_max_length)

generation_config = GenerationConfig(
    max_new_tokens=decoder_max_length,
    early_stopping=True,
    num_beams=5,
    forced_bos_token_id=2
    )

config = wandb.config          # Initialize config
config.TRAIN_BATCH_SIZE = 60  # input batch size for training 2 - 14 Gb, 32 - 16.5 Gb, 64 - 22.9 Gb
config.VALID_BATCH_SIZE = 10   # input batch size for testing (default: 1000)
config.TRAIN_EPOCHS = 100      # number of epochs to train (default: 10)
config.VAL_EPOCHS = 1 
config.LEARNING_RATE = 1e-5    # learning rate (default: 0.01)
config.SEED = 42               # random seed (default: 42)
config.MAX_SOURCE_LEN = encoder_max_length            # source len
config.MAX_TARGET_LEN = decoder_max_length        # target len
config.N_STEPS = 100

torch.manual_seed(config.SEED) # pytorch random seed
np.random.seed(config.SEED) # numpy random seed

def batch_tokenize_preprocess(batch, tokenizer):
    source, target = batch["ctext"], batch["text"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=config.MAX_SOURCE_LEN
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=config.MAX_TARGET_LEN
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)
# sys.exit()

"""Fine-tune mBART 50"""
model_name = f"{model_checkpoint.split('/')[-1]}-context" 
output_model_dir = f"output_{corpus}/{model_name}"
if not os.path.exists(output_model_dir):
    os.makedirs(output_model_dir)
print('Saving to ', output_model_dir)
generation_config.save_pretrained(output_model_dir, "generation_config.json")

args = Seq2SeqTrainingArguments(
    output_dir=output_model_dir,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=config.N_STEPS,
    save_strategy="epoch",
    do_train=True, 
    do_predict=True, 
    do_eval=True,
    predict_with_generate=True,
    learning_rate=config.LEARNING_RATE,
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VALID_BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    num_train_epochs=config.TRAIN_EPOCHS,
    generation_max_length=config.MAX_TARGET_LEN,
    generation_num_beams=5,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    report_to="wandb", 
    run_name=f"{corpus}_mbart50_sent",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=tokenizer.pad_token_id)

metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id) # predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores (evaluate library returns plain floats)
    result = {key: value * 100 for key, value in result.items() if isinstance(value, float)}

    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Function that returns an untrained model to be trained
def model_init():
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    wandb.watch(model, log="all")
    model.config.max_length = config.MAX_TARGET_LEN
    return model

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print('Start training')
trainer.train()
trainer.save_model()
