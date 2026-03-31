import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import transformers
import os
import pandas as pd

from transformers import GenerationConfig
from time import perf_counter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_id_base = "meta-llama/Meta-Llama-3.1-8B"

test_data = 'corpus_codiesp/test_system_context_rag_8.tsv'
df_test = pd.read_csv(test_data, sep='\t', dtype=str)


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3.1-8B"
output_dir = f"./output_codiesp/{model_id.split('/')[-1]}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id_base)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

prompt_template="""Genera la definición del siguiente término médico en la Clasificación Internacional de Enfermedades CIE-10: {context}"""
# cómo usarlo con un LLM:
system_prompt = "Eres un asistente médico útil, responsable y honesto. Eres experto en la Clasificación Internacional de Enfermedades CIE-10 y puedes generar definiciones de términos médicos."
tokenizer.chat_template = prompt_template

responces = []
for i, row in df_test.iterrows():

    context = row['input_with_entity'] + ' ' + row['context'] + ' ' + row['rag']
        
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt_template.format(context=context)}
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=128,
    )
    print('Response:')
    # print(outputs[0]["generated_text"][-1])
    responces.append(outputs[0]["generated_text"][-1]['content'])


df_test['pred'] = responces
df_test.to_csv(f'{output_dir}/test_predictions_rag8_pipeline.tsv', sep='\t', index=False)