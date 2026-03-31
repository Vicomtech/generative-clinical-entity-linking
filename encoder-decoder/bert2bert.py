import numpy as np
import pandas as pd 
import os 
import sys
# import nlp

import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

from transformers import (
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    XLMRobertaTokenizer,
    EncoderDecoderModel,
    AutoTokenizer,
    EvalPrediction, 
    GenerationConfig
)

import evaluate

from datasets import load_dataset
from rouge import Rouge 
import wandb

rouge = evaluate.load("rouge")


# gpu_ids = ["0"]
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
os.environ['TOKENIZERS_PARALLELISM']= "false"
cache_dir = os.environ.get('CACHE_FOLDER') or os.environ.get('HF_HOME') or None
if cache_dir:
    os.environ['HF_HOME'] = cache_dir

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
seed = 999

## LOAD DATASET
data_path = os.path.join('corpus', 'codiesp')
data = os.path.join(data_path, 'train_sentences_words_icd10.tsv')

df = pd.read_csv(data, sep='\t')
print('CORPUS', len(df))
df = df.drop_duplicates(subset=['label', 'descripcion', 'words'], keep='first')
print('UNIQE', len(df))

df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True) # necessary for dataloader

n = round(len(df)/10)
print('DEV split', n)

train_df = df.iloc[n:]
eval_df = df.iloc[:n]
eval_df.reset_index(drop=True, inplace=True)

data_train = 'train_c_u.tsv'
data_eval = 'eval_c_u.tsv'

# sys.exit()

train_df.to_csv(data_train, index=False, sep='\t')
eval_df.to_csv(data_eval, index=False, sep='\t')

dataset = load_dataset('csv', data_files={'train': data_train, 'test': data_eval}, delimiter='\t')

dataset.remove_columns('ids')
dataset.remove_columns('label_type')
dataset.remove_columns('label')
dataset.remove_columns('offset')

print(dataset)

train_dataset = dataset['train']
print('TEXTS FOR TRAINING: ', len(train_dataset))
val_dataset = dataset['test']
print('TEXTS FOR EVALUATING: ', len(val_dataset))

## MODEL
model_output_folder = f"./outputs_bert2bert_codiesp_mbert_uniq-{seed}"
model_name = "bert-base-multilingual-uncased"
# model_name = "BSC-TeMU/roberta-base-biomedical-clinical-es"
# model_name = "dccuchile/bert-base-spanish-wwm-uncased"
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

def calculate_max_tokens(texts):
   token_lengths = [len(tokenizer.encode(text)) for text in texts]
   max_tokens = max(token_lengths)
   if max_tokens % 2 != 0:
        max_tokens += 1
   return max_tokens

input_max_length = calculate_max_tokens(train_df.words.to_list())
print('Max input tokens:', input_max_length)
output_max_length = calculate_max_tokens(train_df.descripcion.to_list())
print('Max output tokens:', output_max_length)
output_min_length = 6
# CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token
# SEP token will work as EOS token
tokenizer.eos_token = tokenizer.sep_token  
tokenizer.pad_token = tokenizer.eos_token

# set decoding params
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

tokenizer.model_max_length = output_max_length

generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    length_penalty = 1.0,
    decoder_start_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    # bos_token_id=tokenizer.bos_token_id,
    pad_token=tokenizer.pad_token_id,
    max_length = output_max_length, 
    min_length = output_min_length,
)

generation_config.save_pretrained(model_output_folder, "generation_config.json")

# set batch size here
batch_size = 24
num_epochs = 200
steps_in_batch =round(len(train_dataset)/batch_size)
print('Number of steps:', steps_in_batch)

# seeds = [42, 123, 987]
# for seed in seeds: 
wandb.init(project="CodiEsp_bert2bert")
training_args = dict(
    output_dir=model_output_folder,
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,
    predict_with_generate = True,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    do_train=True,
    do_eval=True,
    do_predict=True,
    logging_steps=100,
    num_train_epochs=num_epochs,
    warmup_steps=1000,
    save_total_limit=1,
    report_to = 'wandb', 
    dataloader_num_workers = 10, 
    load_best_model_at_end = True, 
    generation_num_beams = 4,
    greater_is_better = True, 
    metric_for_best_model = 'eval_rougeL_f1', 
    seed = seed, 
    generation_config=generation_config
)

training_args = Seq2SeqTrainingArguments(**training_args)


def map_to_encoder_decoder_inputs(examples):
  model_inputs = tokenizer(
      examples["words"].strip(),
      max_length=input_max_length,
      padding='max_length',
      truncation=True,
  )
  targets = tokenizer(
      examples["descripcion"].strip(),
      max_length=output_max_length,
      padding='max_length',
      truncation=True,
  )
  model_inputs["labels"] = targets["input_ids"]
  model_inputs["labels"] = model_inputs["labels"][1:]

  return model_inputs


def create_hf_dataset(dataset):
  with training_args.main_process_first(desc="dataset map pre-processing"):
    dataset = dataset.map(
      map_to_encoder_decoder_inputs,
      remove_columns=dataset.column_names,
    )
    dataset.add_column("length", [len(input_ids) for input_ids in dataset["input_ids"]])
  return dataset

def compute_metrics(pred: EvalPrediction):
    pred_ids = pred.predictions
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    rouge_output2 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"], use_aggregator=True)["rouge2"]
    rouge_output1 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"], use_aggregator=True)["rouge1"]
    rouge_outputL = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"], use_aggregator=True)["rougeL"]
    result = {
        "rouge1_f1": round(rouge_output1, 4)*100,
        "rouge2_f1": round(rouge_output2, 4)*100,
        "rougeL_f1": round(rouge_outputL, 4)*100,
    }
    print(result)
    return result


train_dataset = create_hf_dataset(train_dataset)
val_dataset = create_hf_dataset(val_dataset)
print(val_dataset)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

trainer.train()

