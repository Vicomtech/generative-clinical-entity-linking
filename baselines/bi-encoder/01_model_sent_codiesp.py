import numpy as np
import pandas as pd
import os
import sys
import pickle
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig, set_seed

# Allow running this file directly from the repo root or nested directories.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.faiss_utils import faiss_search
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


## OPTIONS
model_name = 'cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
set_seed(42)

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config).to(device)


def texts2vectors(text_list, model=model, tokenizer=tokenizer, device=device): 
    encoded_input = tokenizer.batch_encode_plus(
        text_list, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)

    torch.cuda.empty_cache()
    return model_output, encoded_input['attention_mask']



def cls_pooling(model_output):
    return model_output[0][:,0]

def get_chunks(ids, n): 
    return [ids[i:i+n] for i in range(0,len(ids),n)]

def flatten(chunks): 
    return [item for sublist in chunks for item in sublist]


output_dir = f'ICD-10-CodiEsp-{model_name.split("/")[-1]}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### DATA
train_dir = 'corpus/codiesp'
test_file = 'test_sentence_description.tsv'
df = pd.read_csv(os.path.join(train_dir, test_file), sep='\t').fillna('')

icd_codes_dir = 'icd_10_es'
df_icd_diag = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_diagnosticos_2020.tsv'), sep='\t', dtype='str')
df_icd_diag = df_icd_diag[["codigo", "descripcion"]]
print('ICD-10 DIAGNÓSTICOS', len(df_icd_diag))

df_icd_proc = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_procedimientos_2020.tsv'), sep='\t', dtype='str') 
df_icd_proc = df_icd_proc[["codigo", "descripcion"]]
print('ICD-10 PROCEDIMIENTOS', len(df_icd_proc))

frames = [df_icd_proc, df_icd_diag]
df_icds = pd.concat(frames)
print('ICD-10:',  len(df_icds))
print()
print("=========")
print()

# Encode
label_types = ['DIAGNOSTICO', 'PROCEDIMIENTO'] 
k = 128   # number of nearest neighbors
pooling = 'cls' 
chunk_size = 4000

dfs = []
for label_type in label_types: 
    df_corpus = df[df['label_type'] == label_type]
    print('Starting with', label_type.upper())

    if label_type == 'DIAGNOSTICO': 
        descriptions = df_icd_diag.descripcion.str.lower().to_list()
        icd_codes = df_icd_diag.codigo.str.lower().to_list()
    elif label_type == 'PROCEDIMIENTO':
        descriptions = df_icd_proc.descripcion.str.lower().to_list()
        icd_codes = df_icd_proc.codigo.str.lower().to_list()

    if df_corpus.empty:
        print('No rows for', label_type, 'in input file. Skipping.')
        continue

    labels = df_corpus.label.str.lower().to_list()
    queries = df_corpus.words.str.lower().to_list()

    output_file_corpus = os.path.join(output_dir, f"{label_type}-{test_file.split('.')[0]}.pkl")

    if os.path.exists(output_file_corpus): 
        print('Corpus embeddings exist, loading')
        with open(output_file_corpus, 'rb') as handle:
            query_embeddings = pickle.load(handle)
    else: 
        print('Calculating corpus embeddings')
        queries_chunks = get_chunks(queries, chunk_size)
        print('Corpus queries chunks', len(queries_chunks))

        query_embeddings_chunks = []
        for i, chunk_queries in enumerate(queries_chunks): 
            print('Corpus chunk', i)

            model_output_queries, attention_mask = texts2vectors(chunk_queries)
            if pooling == 'mean':
                query_embeddings0 = mean_pooling(model_output_queries, attention_mask)
            elif pooling == 'cls':
                query_embeddings0 = cls_pooling(model_output_queries)
            query_embeddings0 = query_embeddings0.cpu().detach().numpy()
            query_embeddings_chunks.append(query_embeddings0)

        query_embeddings = flatten(query_embeddings_chunks)
        # print(query_embeddings[0])
        with open(output_file_corpus, 'wb') as handle:
            pickle.dump(query_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    output_file_icd10 = os.path.join(output_dir, f'{label_type}-icd10.pkl')

    if os.path.exists(output_file_icd10): 
        print('Database emdeddings exist, loading')
        with open(output_file_icd10, 'rb') as handle:
            descriptions_embeddings = pickle.load(handle)
        print('Database embeddings LOADED', len(descriptions_embeddings))
    else: 
        print('Calcualting database embeddings')
        descriptions_chunks = get_chunks(descriptions, chunk_size)
        
        print('Database chunks', len(descriptions_chunks))

        descriptions_embeddings_chunks = []
        for i, chunk_descr in enumerate(descriptions_chunks):  
            print('Descriptions chunk ', i)
            model_output_descr, attention_mask = texts2vectors(chunk_descr)
            if pooling == 'mean':
                descriptions_embeddings0 = mean_pooling(model_output_descr, attention_mask)
            elif pooling == 'cls':
                descriptions_embeddings0  = cls_pooling(model_output_descr)
            descriptions_embeddings0 = descriptions_embeddings0.cpu().detach().numpy()
            descriptions_embeddings_chunks.append(descriptions_embeddings0)

        descriptions_embeddings = flatten(descriptions_embeddings_chunks)
        with open(output_file_icd10, 'wb') as handle:
            pickle.dump(descriptions_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    d = len(descriptions_embeddings[0])

    D, I = faiss_search(descriptions_embeddings, query_embeddings, k=k, d=d)

    result_dicts = []
    for inds, ds in zip(I, D): # inds in ICD-10 corpus
        top_descriptions = '|'.join([descriptions[i] for i in inds])
        top_predictions = '|'.join([icd_codes[i] for i in inds])
        result_dicts.append({'top_descriptions': top_descriptions, 'distancies': ds, 'top_codes': top_predictions})

    df_res = pd.DataFrame(result_dicts)
    df_corpus = df_corpus.copy()
    df_corpus.loc[:, 'top_descriptions']  = df_res['top_descriptions'].values
    df_corpus.loc[:, 'top_codes'] = df_res['top_codes'].values
    df_corpus.loc[:, 'distancies'] = df_res['distancies'].values
    dfs.append(df_corpus)

if not dfs:
    raise ValueError('No predictions generated. Check input labels and corpus files.')

df_results = pd.concat(dfs, axis=0)
df_results = df_results.sort_index()
df_results.to_csv(os.path.join(output_dir, "sentbert_result.tsv"), sep='\t', index=False)
