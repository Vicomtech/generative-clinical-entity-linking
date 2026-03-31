import numpy as np
import pandas as pd 
# from pandas.core.common import SettingWithCopyWarning

import os
import warnings
warnings.simplefilter(action="ignore")

from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer, GenerationConfig
from datasets import load_dataset, Dataset, DatasetDict

from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50Tokenizer

from eval.evaluation_utils import clean_text, calculate_max_tokens, calculate_rouge, calculate_accuracy, calculate_bertscore, calculate_bleu, calculate_meteor
from eval.evaluation_utils import calculate_semscore

from dotenv import load_dotenv
load_dotenv()


## DATA
corpus = 'codiesp'
data_path = os.path.join('corpus', corpus, 'sentence2description_t5')
print(data_path)
data_train = os.path.join(data_path, 'train_sentence2description_t5.tsv')
data_dev = os.path.join(data_path, 'dev_sentence2description_t5.tsv')
data_test = os.path.join(data_path, 'test_sentence2description_t5.tsv')

train_df = pd.read_csv(data_train, sep='\t', dtype='str').fillna('')
train_df = train_df.sample(frac=1, random_state=42)
train_df.reset_index(drop=True, inplace=True) # necessary for dataloader

eval_df = pd.read_csv(data_dev, sep='\t', dtype='str').fillna('')
eval_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(data_test, sep='\t', dtype='str').fillna('')
test_df.reset_index(drop=True, inplace=True)

prefix = "Genera una definición para el término entre <label>: "
mention_column = 'sentence'
description_column = 'description'

# print(train_df.head())
train_df = train_df.rename(columns={mention_column: "ctext", description_column: "text"})
print(train_df.head())
train_df = train_df[['ctext', 'text']]
train_df.ctext = prefix + train_df.ctext
train_df.to_csv('train.tsv', sep='\t', index=False)
# print(train_df.head())
print(len(train_df))

eval_df = pd.read_csv(data_dev, sep='\t', dtype='str').fillna('')
eval_df.reset_index(drop=True, inplace=True)
eval_df = eval_df.rename(columns={mention_column: "ctext", description_column: "text"})
eval_df = eval_df[['ctext', 'text']]
eval_df.ctext = prefix + eval_df.ctext
eval_df.to_csv('dev.tsv', sep='\t', index=False)
print(eval_df.head())

test_df = pd.read_csv(data_test, sep='\t', dtype='str').fillna('')
test_df.reset_index(drop=True, inplace=True)
test_df0 = test_df.rename(columns={mention_column: "ctext", description_column: "text"})
test_df = test_df0[['ctext', 'text']]
# test_df.ctext = prefix + eval_df.ctext
test_df.to_csv('test.tsv', sep='\t', index=False)
print(eval_df.head())

df_base = eval_df 
eval_file = 'dev.tsv'

part = 'test'
if part == 'test': 
    print('PART to evaluate: ', part.upper())
    df_base = test_df
    eval_file = 'test.tsv'

dataset = load_dataset('csv', data_files={'train': 'train.tsv', 'test': eval_file}, delimiter='\t')

input_column = "ctext" 
reference_column = 'text'   

train_dataset = dataset['train']
test_dataset = dataset['test']

checkpoint = 12000
eval_model_dir = f'output_{corpus}/mbart-large-50-context/checkpoint-{checkpoint}'

tokenizer = MBart50Tokenizer.from_pretrained(eval_model_dir)
model = MBartForConditionalGeneration.from_pretrained(eval_model_dir)
model.to("cuda")

generation_config = GenerationConfig.from_pretrained(eval_model_dir, "generation_config.json")

model_name = eval_model_dir.split('/')[-1]
batch_size = 64

def calculate_max_tokens(texts):
   token_lengths = [len(tokenizer.encode(text)) for text in texts]
   max_tokens = max(token_lengths)
   if max_tokens % 2 != 0:
        max_tokens += 1
   return max_tokens + 2

input_max_length = calculate_max_tokens(train_df.ctext.to_list())
output_max_length = calculate_max_tokens(train_df.text.to_list())
output_min_length = 6

def generate_descriptions(batch):
    inputs = tokenizer(batch[input_column], truncation=True, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    outputs = model.generate(
        input_ids, 
        num_return_sequences=1, 
        num_beams=5, 
        length_penalty=1.2, 
        temperature=0.6,
        generation_config=generation_config, 
        attention_mask=attention_mask
        )
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred"] = output_str
    # print(output_str)
    return batch

results = test_dataset.map(generate_descriptions, batched=True, batch_size=batch_size)

predictions_lst = results["pred"]
predictions_lst_clean = []
for line in predictions_lst: 
    new_line = clean_text(line)
    predictions_lst_clean.append(new_line)

description_list = df_base[reference_column].to_list()

description_list_clean = []
for line in description_list: 
    new_line = clean_text(line)
    description_list_clean.append(new_line)

print('Absolute accuracy: ', calculate_accuracy(predictions_lst_clean, description_list_clean))
print('ROUGE CLEAN: ', calculate_rouge(predictions_lst_clean, description_list_clean))
print('ROUGE: ', calculate_rouge(predictions_lst, description_list))
rouge_score = calculate_rouge(predictions_lst, description_list)

print('BLEU: ', calculate_bleu(predictions_lst, description_list))

print('METEOR: ', calculate_meteor(predictions_lst, description_list))

berscore = calculate_bertscore(predictions_lst, description_list)
print('BERTSCORE: ', berscore)

semscore, sscores =  calculate_semscore(predictions_lst, description_list)
print('SEMSCORE: ', semscore)

se_descrs = pd.Series(description_list_clean)
se_preds = pd.Series(predictions_lst_clean)
se_results = pd.Series(results["pred"])

# df_base['description_clean'] = se_descrs.values
# df_base['bart50_clean'] = se_preds.values
test_df0['pred'] = se_results.values
test_df0['semscore'] = sscores
test_df0.to_csv(os.path.join(eval_model_dir, part+'_predictions.tsv'), sep='\t', index=False)

df_scores = pd.DataFrame({
    'ACCURACY': [calculate_accuracy(predictions_lst, description_list)],
    'ROUGE-L F1': [rouge_score['rougeL_f1']], 
    'BLEU': [calculate_bleu(predictions_lst, description_list)],
    'METEOR': [calculate_meteor(predictions_lst, description_list)],
    'BERTSCORE': [berscore],
    'SEMSCORE': [semscore]
    })
df_scores.to_csv(os.path.join(eval_model_dir, part+'_scores.tsv'), sep='\t', index=False)

df_unseen = test_df[(test_df['ctext'].isin(train_df['ctext'].values)) & (test_df['ctext'].isin(eval_df['ctext'].values))]
print('Unseen data: ', len(df_unseen))
