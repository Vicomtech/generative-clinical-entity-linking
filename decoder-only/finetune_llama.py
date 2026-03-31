## https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795

import os
import sys
import pandas as pd

import torch
import logging 

import wandb
logging.getLogger("wandb").setLevel(logging.ERROR)

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
cache_folder = os.environ.get('CACHE_FOLDER') or os.environ.get('HF_HOME') or None

model_id="meta-llama/Meta-Llama-3.1-8B"

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                              add_bos=True, 
                                              add_eos=True, 
                                              padding_side="right",
                                              cache_dir=cache_folder)
    print(tokenizer.eos_token)
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.pad_token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_folder, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

## DATA 
corpus = "codiesp"
train_file = f"corpus_{corpus}/train_system_context.tsv"
dev_file = f"corpus_{corpus}/dev_system_context.tsv"

df_tr = pd.read_csv(train_file, sep='\t', dtype=str)
df_d = pd.read_csv(dev_file, sep='\t', dtype=str)

corpus_mapped = 'mapped'
df_train_mapped = pd.read_csv(f"corpus_{corpus_mapped}/train_system_instruction_context.tsv", sep='\t', dtype=str)
df_dev_mapped = pd.read_csv(f"corpus_{corpus_mapped}/dev_system_instruction_context.tsv", sep='\t', dtype=str)

df_train = pd.concat([df_tr, df_train_mapped])
df_dev = pd.concat([df_d, df_dev_mapped])
df_train = df_tr
df_dev = df_d
print('Train:', len(df_train))
print('Dev:', len(df_dev))

dataset_train = Dataset.from_pandas(df_train, split="train")
dataset_dev = Dataset.from_pandas(df_dev, split="dev")

def formatted_train(system, inp, response)->str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

def prepare_train_data(data_df):
    data_df['input_context'] = data_df['instruction'] + ' ' + data_df['input_with_entity'] + '; contexto: ' + data_df['context']
    data_df["text"] = data_df[["system", "input_context", "output"]].apply(lambda x: formatted_train(x['system'], x['input_context'], x['output']), axis=1)
    data = Dataset.from_pandas(data_df)
    return data

dataset_train = prepare_train_data(df_train)
dataset_dev = prepare_train_data(df_dev)

print('Example:')
print(dataset_train['text'][0])

# sys.exit()

peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

wandb.init(project=f"{corpus}-LLM-instruct", # the project I am working on
           job_type="train",
           tags=["hf_sft_lora"]) # the Hyperparameters I want to keep track of

output_model=output_dir = f"./output_{corpus}/{model_id.split('/')[-1]}-system-instruct-context"

sft_config = SFTConfig(
    output_dir=output_model,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    do_eval=True,
    eval_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_steps=10,
    num_train_epochs=100,
    fp16=True,
    report_to="wandb",
    dataset_text_field="text",
    max_seq_length=512
)

trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        peft_config=peft_config,
        args=sft_config,
        packing=False,
    )

trainer.train()


# training_arguments = TrainingArguments(
#         output_dir=output_model,
#         per_device_train_batch_size=36,
#         gradient_accumulation_steps=16,
#         optim="paged_adamw_32bit",
#         learning_rate=2e-4,
#         lr_scheduler_type="cosine",
#         save_strategy="epoch",
#         do_eval=True,
#         eval_strategy="epoch",
#         save_total_limit=1,
#         load_best_model_at_end=True,
#         logging_steps=100,
#         num_train_epochs=50,
#         fp16=True,
#         report_to="wandb",
#     )