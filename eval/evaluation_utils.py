
import pandas as pd 

import os
import warnings
warnings.simplefilter(action="ignore")

from dotenv import load_dotenv
load_dotenv()

import torch
import evaluate
import bert_score.utils as _bsu

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


_BERTSCORE_SAFE_MAX = 512
_orig_sent_encode = _bsu.sent_encode

def _safe_sent_encode(tokenizer, a):
    if getattr(tokenizer, 'model_max_length', _BERTSCORE_SAFE_MAX) > 100_000:
        tokenizer.model_max_length = _BERTSCORE_SAFE_MAX
    return _orig_sent_encode(tokenizer, a)

_bsu.sent_encode = _safe_sent_encode


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
cache_folder = os.environ.get("CACHE_FOLDER") or os.environ.get("HF_HOME") or None

model_id = 'sentence-transformers/distiluse-base-multilingual-cased-v1' # 2

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_folder)
model = AutoModel.from_pretrained(model_id, cache_dir=cache_folder)


model.to('cuda')

def calculate_semscore(list1, list2, model=model):
    # Tokenize sentences
    sscores = []
    for sentence1, sentence2 in zip(list1, list2):
        sentence1 = str(sentence1)
        sentence2 = str(sentence2)
        encoded_input = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors='pt').to('cuda')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p = 2, dim = 1)

        sscore = (sentence_embeddings[0] @ sentence_embeddings[1]).item()
        sscores.append(sscore)

    score = round(sum(sscores)/len(sscores)*100, 2)
    return score, sscores

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

def replace_diacritica(text, diacritica=diacritica): 
    for letter, replacement in diacritica.items():
        text = text.replace(letter, replacement)
    return text

stop = ['!', '"', '$', '%', '&', "'", '€', '´', '：', '¨',
               ')', '*', '+', ',', '-', '.', '•',
               '/', ':', ';', '<', '=', '>', '?', '∩', '£',
               '[', '\\', ']', '^', '_', '`', '', '@', '#',
               '{', '|', '}', '~', '–', '—', '"', '■',
               "¿", "¡", "''", "...", '_', '´', '♪',
               '“', '”', '…', '‘', "'", "``", '„', '’',
               '°', '«', '»', '×', '》》', 'ʖ', '(']

def clean_text(line, diacritica=diacritica): 
    for i in stop: 
        line = str(line)
        line = line.replace(i, '')
        line = ' '.join(line.split()) 
        line = replace_diacritica(line, diacritica)
        line = line.lower()
    return line

def calculate_max_tokens(texts, tokenizer):
   token_lengths = [len(tokenizer.encode(text)) for text in texts]
   max_tokens = max(token_lengths)
   if max_tokens % 2 != 0:
        max_tokens += 1
   return max_tokens


def get_icd_code(preds, icd_df): 
    # get icd code from predicted description
    icd_codes = []
    icd_df['descripcion'] = icd_df['descripcion'].apply(clean_text)
    preds = [clean_text(pred) for pred in preds]
    for pred in preds: 
        icd_code_df = icd_df[icd_df['descripcion'] == pred]
        # print(icd_code_df)
        if icd_code_df.shape[0] == 0: 
            icd_code = 'no_code'
        else:
            icd_code = icd_code_df['codigo'].values[0]
        icd_codes.append(icd_code)
    # print(icd_codes)
    return icd_codes

def calculate_code_accuracy(preds, refs): 
    marks = []
    for p, t in zip(preds, refs):
        if p.upper() == t.upper(): 
            marks.append(1)
        else: 
            marks.append(0)
    acc = round(sum(marks)/len(preds)*100, 2)
    return acc, marks

rouge = evaluate.load("rouge")
def calculate_rouge(preds, refs): 
    rouge_output2 = rouge.compute(predictions=preds, references=refs, rouge_types=["rouge2"], use_aggregator=True)["rouge2"]
    rouge_output1 = rouge.compute(predictions=preds, references=refs, rouge_types=["rouge1"], use_aggregator=True)["rouge1"]
    rouge_outputL = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"], use_aggregator=True)["rougeL"]
    rouge_result = {
        "rouge1_f1": round(rouge_output1, 4)*100,
        "rouge2_f1": round(rouge_output2, 4)*100,
        "rougeL_f1": round(rouge_outputL, 4)*100,
    }
    return rouge_result


def compute_rouge_scores(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        rouge_outputL = rouge.compute(predictions=[pred], references=[ref], rouge_types=["rougeL"], use_aggregator=True)["rougeL"]
        scores.append(round(rouge_outputL*100, 2))
    return scores

def calculate_accuracy(preds, refs): 
    marks = []
    for p, t in zip(preds, refs): 
        if p == t: 
            marks.append(1)
        else: 
            marks.append(0)
    acc = sum(marks)/len(preds)
    acc = round(acc*100, 2)
    return acc


# model_name = 'roberta-large-mnli'
model_name = "microsoft/deberta-xlarge-mnli"
bertscore = evaluate.load("bertscore")

def calculate_bertscore(preds, refs, model=model_name):
    results_bertscore = bertscore.compute(
            predictions=preds,
            references=refs,
            model_type=model,
            lang="es")

    f1s = results_bertscore['f1']
    f1_bertscore = round(sum(f1s)/len(f1s)*100, 2)
    return f1_bertscore, f1s


bleu = evaluate.load("bleu")
def calculate_bleu(preds, list_refs): 
    result = bleu.compute(predictions=preds, references=list_refs)
    return round(result['bleu']*100, 2)

def compute_bleu_scores(predictions, references): 
    scores = []
    for pred, ref in zip(predictions, references):
        if len(pred) == 0: 
            pred = 'no'   
        result = bleu.compute(predictions=[pred], references=[ref])
        scores.append(round(result['bleu']*100, 2))
    return scores

meteor = evaluate.load('meteor')

def calculate_meteor(preds, refs): 
    results = meteor.compute(predictions=preds, references=refs)
    return round(results['meteor']*100, 2)

def compute_meteor_scores(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        result = meteor.compute(predictions=[pred], references=[ref])
        scores.append(round(result['meteor']*100, 2))
    return scores

def calculate_all_scores(predictions_lst, description_list):
    description_list_clean = []
    for line in description_list:
        new_line = clean_text(line)
        description_list_clean.append(new_line)

    predictions_list_clean = []
    for line in predictions_lst:
        new_line = clean_text(line)
        predictions_list_clean.append(new_line)

    accuracy = calculate_accuracy(predictions_list_clean, description_list_clean)
    rouge = calculate_rouge(predictions_lst, description_list)
    bleu = calculate_bleu(predictions_lst, description_list)
    meteor = calculate_meteor(predictions_lst, description_list)

    bertscore, bertscores = calculate_bertscore(predictions_lst, description_list)
    semscore, sscores = calculate_semscore(predictions_lst, description_list) 
    df_scores = pd.DataFrame({
        'ACCURACY': [accuracy],
        'ROUGE-L-F1': [rouge['rougeL_f1']], 
        'BLEU': [bleu],
        'METEOR': [meteor],
        'BERTSCORE': [bertscore],
        'SEMSCORE': [semscore]
        })
    
    df_list_scores = pd.DataFrame({
        'BERTSCORE': bertscores,
        'SEMSCORE': sscores, 
    })
    return df_scores, df_list_scores

def calculate_bleu_meter_rouge(preds, refs): 
    bleu = compute_bleu_scores(preds, refs)
    meteor = compute_meteor_scores(preds, refs)
    rouge = compute_rouge_scores(preds, refs)
    df = pd.DataFrame({
        'BLEU': bleu,
        'METEOR': meteor,
        'ROUGE-L-F1': rouge
    })
    return df


