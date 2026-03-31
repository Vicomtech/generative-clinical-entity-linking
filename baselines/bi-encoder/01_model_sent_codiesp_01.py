import numpy as np
import pandas as pd
import torch
import time
import os
import pickle
import sys
from pathlib import Path
from sentence_transformers import util

# Allow running this file directly from the repo root or nested directories.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.faiss_utils import faiss_search
from eval.transformer_utils import texts2vectors, cls_pooling, get_chunks, flatten, mean_pooling

from sklearn.metrics import f1_score


## OPTIONS
# model_name = 'FremyCompany/BioLORD-2023-M'
model_name = 'cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large'
# model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
# model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
# model_name  = 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'

add_train = 'term_only' # 'term_only', 'add_train' , 'train_only'
print(f"Model: {model_name}, database: {add_train}")


output_dir = f'ICD-10-CodiEsp-{model_name.split("/")[-1]}-{add_train}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### DATA
train_dir = 'corpus/codiesp'
test_file = 'test_sentence_description.tsv'
df = pd.read_csv(os.path.join(train_dir, test_file), sep='\t').fillna('')
df['words'] = df['words'].str.lower()


df_train = pd.read_csv(os.path.join(train_dir, 'train_sentence_description.tsv'), sep='\t').fillna('')
df_train = df_train.drop_duplicates(subset=['words'])
df_train['words'] = df_train['words'].str.lower()
df_train = df_train[["label", "words", "label_type"]]

df_unseen = df[~df['words'].isin(df_train['words'].values)]

print('UNSEEN', len(df_unseen))

df = df_unseen

icd_codes_dir = 'icd_10_es'
df_icd_diag = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_diagnosticos_2020.tsv'), sep='\t', dtype='str')
df_icd_diag = df_icd_diag[["codigo", "descripcion"]]
print('ICD-10 DIAGNOSTICOS', len(df_icd_diag))
if add_train == 'add_train':
    df_train_d = df_train[df_train['label_type'] == 'DIAGNOSTICO']
    df_train_d = df_train_d.rename(columns={"label": "codigo", "words": "descripcion"})
    df_train_d = df_train_d[["codigo", "descripcion"]] 
    df_icd_d = pd.concat([df_icd_diag, df_train_d], axis=0, ignore_index=True)
    print('ICD-10 DIAGNOSTICOS + TRAIN', len(df_icd_d))
elif add_train == 'train_only':
    df_icd_d = df_train[df_train['label_type'] == 'DIAGNOSTICO']
    df_icd_d = df_icd_d.rename(columns={"label": "codigo", "words": "descripcion"})
    df_icd_d = df_icd_d[["codigo", "descripcion"]] 
    print('TRAIN DIAGNOSTICOS', len(df_icd_d))
else: 
    df_icd_d = df_icd_diag

df_icd_proc = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_procedimientos_2020.tsv'), sep='\t', dtype='str') 
df_icd_proc = df_icd_proc[["codigo", "descripcion"]]
print('ICD-10 PROCEDIMIENTOS', len(df_icd_proc))
if add_train == 'add_train':
    df_train_p = df_train[df_train['label_type'] == 'PROCEDIMIENTO']
    df_train_p = df_train_p.rename(columns={"label": "codigo", "words": "descripcion"})
    df_train_p = df_train_p[["codigo", "descripcion"]]
    df_icd_p = pd.concat([df_icd_proc, df_train_p], axis=0, ignore_index=True)
    print('ICD-10 PROCEDIMIENTOS + TRAIN', len(df_icd_p))
elif add_train == 'train_only':
    df_icd_p = df_train[df_train['label_type'] == 'PROCEDIMIENTO']
    df_icd_p = df_icd_p.rename(columns={"label": "codigo", "words": "descripcion"})
    df_icd_p = df_icd_p[["codigo", "descripcion"]]
    print('TRAIN PROCEDIMIENTOS', len(df_icd_p))
else: 
    df_icd_p = df_icd_proc

frames = [df_icd_p, df_icd_d]
df_icds = pd.concat(frames)
print('ONTOLOGÍA:',  len(df_icds))

print()
print("=========")
print()

# Encode
label_types = ['DIAGNOSTICO', 'PROCEDIMIENTO'] 
k = 128   # number of nearest neighbors
pooling = 'mean' # 'mean' or 'cls'
if model_name == 'cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large':
    pooling = 'cls' 
chunk_size = 2000

dfs = []
for label_type in label_types: 
    df_corpus = df[df['label_type'] == label_type]
    print('Starting with', label_type.upper())

    if label_type == 'DIAGNOSTICO': 
        descriptions = df_icd_d.descripcion.str.lower().to_list()
        icd_codes = df_icd_d.codigo.str.lower().to_list()
    elif label_type == 'PROCEDIMIENTO':
        descriptions = df_icd_p.descripcion.str.lower().to_list()
        icd_codes = df_icd_p.codigo.str.lower().to_list()

    labels = df_corpus.label.str.lower().to_list()
    queries = df_corpus.words.str.lower().to_list()

    output_file_corpus = os.path.join(output_dir, f"{label_type}-{test_file.split('.')[0]}.pkl")

    if os.path.exists(output_file_corpus):
        print('Corpus embeddings exist, loading')
        try:
            query_embeddings = pickle.load(open(output_file_corpus, "rb"))
        except (EOFError, pickle.UnpicklingError):
            print('Corpus cache corrupted, regenerating')
            os.remove(output_file_corpus)
    if not os.path.exists(output_file_corpus):
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
        print('Database embeddings exist, loading')
        try:
            descriptions_embeddings = pickle.load(open(output_file_icd10, "rb"))
            print('Database embeddings LOADED', len(descriptions_embeddings))
        except (EOFError, pickle.UnpicklingError):
            print('Database cache corrupted, regenerating')
            os.remove(output_file_icd10)
    if not os.path.exists(output_file_icd10):
        print('Calculating database embeddings')
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
    for inds, ds, vecs, label in zip(I, D, query_embeddings, labels): #inds in ICD-10 corpus
        top_descriptions = '|'.join([descriptions[i] for i in inds])
        top_predictions = '|'.join([icd_codes[i] for i in inds])
        result_dicts.append({'top_descriptions': top_descriptions, 'distancies': ds, 'top_codes': top_predictions})

    df_res = pd.DataFrame(result_dicts)
    df_corpus = df_corpus.copy()
    df_corpus.loc[:, 'top_descriptions']  = df_res['top_descriptions'].values
    df_corpus.loc[:, 'top_codes'] = df_res['top_codes'].values
    df_corpus.loc[:, 'distancies'] = df_res['distancies'].values
    dfs.append(df_corpus)

df_results = pd.concat(dfs, axis=0)
df_results = df_results.sort_index()
df_results.to_csv(os.path.join(output_dir, "sentbert_result.tsv"), sep='\t', index=False)
