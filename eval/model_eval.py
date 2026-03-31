import pandas as pd 
# from pandas.core.common import SettingWithCopyWarning

import os
import warnings
warnings.simplefilter(action="ignore")

from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer, GenerationConfig
from datasets import load_dataset

import evaluate

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

diacritica = {
	"á": "a",
	"ó": "o",
	"í": "i",
	"é": "e",
	"ú": "u",
	"ü": "u",
	"ù": "u",
	"à": "a",
	"è": "e",
	"ï": "i",
	"ò": "o", 
    "ñ": "n", 
    "ç": "c"
}

def replaceDiacritica(text, diacritica): 
    for letter, replacement in diacritica.items():
        text = text.replace(letter, replacement)
    return text

stop = ['!', '"', '$', '%', '&', "'", '€', '´', '（', '：', '¨',
               ')', '*', '+', ',', '-', '.', '•',
               '/', ':', ';', '<', '=', '>', '?', '∩', '£',
               '[', '\\', ']', '^', '_', '`', '', '@', '#',
               '{', '|', '}', '~', '–', '—', '"', '■',
               "¿", "¡", "''", "...", '_', '´', '♪',
               '“', '”', '…', '‘', "'", "``", '„', '’',
               '°', '«', '»', '×', '》》', 'ʖ', '(']

def removeStop(line, diacritica): 
    for i in stop: 
        line = line.replace(i, '')
        line = ' '.join(line.split()) 
        line = replaceDiacritica(line, diacritica)
        line = line.lower()
    return line


## LOAD DATASET
data_path = '/DATA/ezotova_data/ICD-10_CodiEsp/data'
test = 'test.tsv'

# test = 'test_l_human_u.tsv'
test_df = pd.read_csv(test, sep='\t', dtype=str)
# test_df = test_df[['filename', 'code', 'text']]
dev = 'dev.tsv'
eval_df = pd.read_csv(dev, sep='\t', dtype=str)

train = 'train.tsv'

part = 'test'

if part == 'eval': 
    print('== PART to evaluate ==', part.upper())
    df_base = eval_df
    dataset = load_dataset('csv', data_files={'train': train, 'test': dev}, delimiter='\t')
elif part == 'test': 
    print('== PART to evaluate ==', part.upper())
    df_base = test_df
    dataset = load_dataset('csv', data_files={'train': train, 'test': test}, delimiter='\t')

input_column = "text" 
reference_column = 'description'   

dataset.remove_columns('filename')
dataset.remove_columns('code')
dataset.remove_columns('description')    

print(dataset)

train_dataset = dataset['train']
(print('TEXTS FOR TRAINING: ', len(train_dataset)))
test_dataset = dataset['test']
(print('TEXTS FOR EVALUATING: ', len(test_dataset)))

eval_model_dir = 'outputs_mbert2mbert_medprocner_main_uniq_42/checkpoint-1757'
model_name = eval_model_dir.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(eval_model_dir,
    cache_dir="/DATA/ezotova_data/python_cache/")

model = EncoderDecoderModel.from_pretrained(eval_model_dir)
model.to("cuda")
generation_config = GenerationConfig.from_pretrained(eval_model_dir, "generation_config.json")

def calculate_max_tokens(texts):
   token_lengths = [len(tokenizer.encode(text)) for text in texts]
   max_tokens = max(token_lengths)
   if max_tokens % 2 != 0:
        max_tokens += 1
   return max_tokens

batch_size = 20
input_max_length = 38
print('Max input tokens:', input_max_length)
output_max_length = 48
print('Max output tokens:', output_max_length)
output_min_length = 6
# CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token
# SEP token will work as EOS token
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
    new_line = removeStop(line, diacritica)
    predictions_lst_clean.append(new_line)

description_list = df_base[reference_column].to_list()

description_list_clean = []
for line in description_list: 
    new_line = removeStop(line, diacritica)
    description_list_clean.append(new_line)

rouge_output2 = rouge.compute(predictions=predictions_lst_clean, references=description_list_clean, rouge_types=["rouge2"], use_aggregator=True)["rouge2"]
rouge_output1 = rouge.compute(predictions=predictions_lst_clean, references=description_list_clean, rouge_types=["rouge1"], use_aggregator=True)["rouge1"]
rouge_outputL = rouge.compute(predictions=predictions_lst_clean, references=description_list_clean, rouge_types=["rougeL"], use_aggregator=True)["rougeL"]
rouge_result = {
    "rouge1_f1": round(rouge_output1, 4)*100,
    "rouge2_f1": round(rouge_output2, 4)*100,
    "rougeL_f1": round(rouge_outputL, 4)*100,
}

print(rouge_result)
print(len(df_base))
print(len(description_list_clean))

se_descrs = pd.Series(description_list_clean)
se_preds = pd.Series(predictions_lst_clean)
se_results = pd.Series(results["pred"])

df_base['description_clean'] = se_descrs.values
df_base['bert2bert_preds_clean'] = se_preds.values
df_base['bert2bert_preds'] = se_results.values

df_base.to_csv(os.path.join(eval_model_dir, model_name+'_'+part+'_predictions.tsv'), sep='\t', index=False)

## ACCURACY 
def accuracy(preds, true): 
    marks = []
    for p, t in zip(preds, true): 
        if p == t: 
            marks.append(1)
        else: 
            marks.append(0)
    acc = sum(marks)/len(preds)
    return acc

acc = accuracy(predictions_lst_clean, description_list_clean)
print('Absolute accuracy: ', round(acc*100, 2))

results_bertscore = bertscore.compute(
    predictions=predictions_lst_clean, 
    references=description_list_clean, 
    model_type="bert-base-multilingual-cased", 
    lang="es")

f1s = results_bertscore['f1']
f1_bertscore = round(sum(f1s)/len(f1s), 4)*100
print('BERTSCORE F1: ', f1_bertscore)
# label_str = results["block_labels"]
# print(label_str)
# texts = results['sentence']

# predictions_set = []
# for i in predictions:
#     i_set = sorted(list(set(i)))
#     predictions_set.append(' '.join(i_set))

# # print(predictions_set[:100])
# # df_test['predictions'] = predictions_set

# true_labels = df_test.block_labels.to_list()

# true_set = []
# for label in true_labels: 
#     label_set = sorted(list(set(label.split())))
#     true_set.append(' '.join(label_set))

# print('SET SCORES')

# print('F1 macro ', f1_score(true_set, predictions_set, average='macro'))

# print('F1 micro ', f1_score(true_set, predictions_set, average='micro'))


# def is_in_true(pred, true): 
#     pred_tok = pred.split()
#     true_tok = true.split()
#     mask = np.isin(pred_tok, true_tok)
#     # print(mask)
#     mark = 0 
#     if True in mask: 
#         mark = 1
#     return mark

# TP = 0
# FP = 0
# FN = 0
# pred_marks = []
# for pred, true in zip(predictions_set, true_set):
#     for c in pred.split():
#         if c in true.split():
#             TP += 1
#         else:
#             FP += 1
#     for c in true.split():
#         if c not in pred.split():
#             FN += 1
#     mark = is_in_true(pred, true)
#     pred_marks.append(mark)

# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# f_score = (2 * precision * recall) / (precision + recall)

# print('NORMAL TP FP FN')
# print('Precision ', precision)
# print('Recall', recall)
# print('F-score all vs all', f_score)
 
# df_test['true_mark'] = 1

# df_test['pred_mark'] = pred_marks

# preds_isin = df_test.pred_mark.to_list()
# true_isin = df_test.true_mark.to_list()

# print('IS IN SCORES P@1')

# print('F1 macro ', f1_score(true_isin, preds_isin, average='macro'))

# print('F1 micro ', f1_score(true_isin, preds_isin, average='micro'))

# print('Accuracy ', accuracy_score(true_isin, preds_isin))

# df_test.to_csv('test_with_preds_12layers.tsv', sep='\t')


