import numpy as np
import pandas as pd 
# from pandas.core.common import SettingWithCopyWarning

import os
import warnings
warnings.simplefilter(action="ignore")

from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer, GenerationConfig
from datasets import load_dataset
import evaluate

from eval.evaluation_utils import clean_text, calculate_max_tokens, calculate_rouge, calculate_accuracy, calculate_bertscore, calculate_bleu, calculate_meteor
from eval.evaluation_utils import calculate_semscore

from dotenv import load_dotenv
load_dotenv()
cache_dir = os.environ.get("CACHE_FOLDER") or os.environ.get("HF_HOME") or None

## LOAD DATASET
data_path = '/DATA/ezotova_data/ICD-10_CodiEsp/data'
test = os.path.join(data_path, 'test_icd2description_20210.tsv')
# test_df = pd.read_csv(test, sep='\t', dtype=str)
# test = 'test_m_main_u.tsv'

# test = 'test_l_human_u.tsv'
test_df = pd.read_csv(test, sep='\t', dtype=str)
# test_df = test_df[['filename', 'code', 'text']]
dev = 'codiesp/eval_c_u.tsv'
eval_df = pd.read_csv(dev, sep='\t', dtype=str)
train = 'codiesp/train_c_u.tsv'
part = 'eval'

eval_file = ''

if part == 'eval': 
    print('== PART to evaluate ==', part.upper())
    df_base = eval_df
    eval_file = 'codiesp/eval_c_u.tsv'

elif part == 'test': 
    print('== PART to evaluate ==', part.upper())
    df_base = test_df
    eval_file = test

dataset = load_dataset('csv', data_files={'train': train, 'test': eval_file}, delimiter='\t')

input_column = "words" 
reference_column = 'descripcion'   

# dataset.remove_columns('filename')
# dataset.remove_columns('code')
# dataset.remove_columns('description')    
dataset.remove_columns('offset')
dataset.remove_columns('descripcion')
dataset.remove_columns('ids')
dataset.remove_columns('label_type')
dataset.remove_columns('label')


train_dataset = dataset['train']
test_dataset = dataset['test']
# test_dataset = test_dataset.select(np.arange(100))
eval_model_dir = 'outputs_bert2bert_codiesp_mbert_uniq_42/checkpoint-4870'
# eval_model_dir = 'outputs_roberta-base-biomedical-clinical-es_codiesp_42/checkpoint-2400'
print(eval_model_dir)
model_name = eval_model_dir.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(eval_model_dir,
    cache_dir=cache_dir)

model = EncoderDecoderModel.from_pretrained(eval_model_dir)
model.to("cuda")
generation_config = GenerationConfig.from_pretrained(eval_model_dir, "generation_config.json")

batch_size = 20
input_max_length = generation_config.max_length
output_max_length = generation_config.max_length
output_min_length = 6
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token  
tokenizer.pad_token = tokenizer.eos_token

# map data correctly
def generate_descriptions(batch):
    inputs = tokenizer(batch[input_column], padding="max_length", truncation=True, max_length=input_max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    outputs = model.generate(
        input_ids, 
        generation_config=generation_config, 
        attention_mask=attention_mask
        )
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred"] = output_str
    return batch

results = test_dataset.map(generate_descriptions, batched=True, batch_size=batch_size)
print(results)

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

print('BLEU CLEAN: ', calculate_bleu(predictions_lst_clean, description_list))
print('BLEU: ', calculate_bleu(predictions_lst, description_list_clean))

print('METEOR CLEAN: ', calculate_meteor(predictions_lst_clean, description_list_clean))
print('METEOR: ', calculate_meteor(predictions_lst, description_list))

print('BERTSCORE CLEAN: ', calculate_bertscore(predictions_lst_clean, description_list_clean))
print('BERTSCORE: ', calculate_bertscore(predictions_lst, description_list))

print('SEMSCORE CLEAN: ', calculate_semscore(predictions_lst_clean, description_list_clean))
print('SEMSCORE: ', calculate_semscore(predictions_lst, description_list))

se_descrs = pd.Series(description_list_clean)
se_preds = pd.Series(predictions_lst_clean)
se_results = pd.Series(results["pred"])

df_base['description_clean'] = se_descrs.values
df_base['preds_clean'] = se_preds.values
df_base['preds'] = se_results.values

df_base.to_csv(os.path.join(eval_model_dir, model_name+'_'+part+'_predictions.tsv'), sep='\t', index=False)



