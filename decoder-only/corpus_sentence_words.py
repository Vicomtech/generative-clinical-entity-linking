import pandas as pd 
import numpy as np
import os
import glob
import re
import sys

# import nltk

import spacy
from spacy.pipeline import Sentencizer
from spacy.lang.es import Spanish

from sklearn.preprocessing import MultiLabelBinarizer

def text_preprocessing(string): 
    string = string.replace('\n', '')
    string = string.replace('.', '. ')
    string = string.replace(',', ', ')
    string = string.replace(';', '; ')
    string = string.replace(':', ': ')
    string = string.replace('?', '? ')
    string = string.replace('!', '! ')
    string = string.replace(')', ') ')
    string = re.sub('\s{2,}', ' ', string)
    return string

def group_labels(row):
    if row['label_type'] == "DIAGNOSTICO":
        val = row['label'].split('.')[0]
        return val
    elif row['label_type'] == "PROCEDIMIENTO":
        if len(row['label']) > 4: 
            val = row['label'][:-4]
        else: 
            val = row['label'][:-1]
        return val
    

def start_end(offset):
    offset_split = offset.split()
    start = offset_split[0]
    end = offset_split[-1]
    return pd.Series([int(start), int(end)])
    
nlp = spacy.load('es_core_news_md')
sentencizer = Sentencizer()
nlp.add_pipe("sentencizer")

# work_dir = os.getcwd()


test_rag_file = os.path.join('..', 'baselines', 'bi-encoder', 'sentbert_result.tsv')
# test_rag_df = pd.read_csv(test_rag_file, sep='\t', dtype=str)

data_dir = os.path.join('..', 'corpus', 'codiesp', 'test')
part_texts_dir = os.path.join(data_dir, 'text_files')

df_set = pd.read_csv(test_rag_file, sep='\t', dtype=str)


text_list = []
for filename in glob.glob(os.path.join(part_texts_dir, '*.txt')):
    fn = filename.split('/')[-1].split('.')[0]
    with open(os.path.join(part_texts_dir, filename), 'r', encoding="utf-8") as f:
        text = f.read()
    
    df_terms = df_set[df_set['ids'] == fn] # check if offsets are correct
    for i, r in df_terms.iterrows():
        start = int(r['offset'].split()[0])
        stop = int(r['offset'].split()[-1])
        # print(start+len(r['words']), stop)
        if (r['words'] != text[start:stop]) and (';' not in r['offset']):
            print(fn)
            print(start+len(r['words']), stop)
            print(r['words'], text[start:stop])
    # text, filename 
    text_list.append([text, fn])
    
sentences = []
offsets = []
file_id = []
for lst in text_list:
    lst[0] = text_preprocessing(lst[0])
    tokenized = nlp(lst[0])
    for s in tokenized.sents:
        # print(s)
        s = str(s)
        off = lst[0].find(s)
        sentences.append(s)
        offsets.append([off, len(s)])
        file_id.append(lst[1].split('/')[-1])   

df_set[['start', 'end']] = df_set['offset'].apply(start_end)

labels_sents = []
labels_sents_group = []
label_types = []
words_sents = []

labs = []
lab_types = []
words = []
sents = []
ids = []
top_descripciones = []
# top_codes = []

for s, o, id_ in zip(sentences, offsets, file_id): 
    s = str(s)
    o = [int(i) for i in o]
    labels_sent = df_set['label'][(df_set['start'] >= o[0]) & (df_set['start'] <= (o[0]+o[1])) & (df_set['ids'] == id_)].to_list()
    label_types = df_set['label_type'][(df_set['start'] >= o[0]) & (df_set['start'] <= (o[0]+o[1])) & (df_set['ids'] == id_)].to_list()
    top_descrs = df_set['top_descriptions'][(df_set['start'] >= o[0]) & (df_set['start'] <= (o[0]+o[1])) & (df_set['ids'] == id_)].to_list()

    # print(labels_sent)
    words_sent = df_set['words'][(df_set['start'] >= o[0]) & (df_set['start'] <= (o[0]+o[1])) & (df_set['ids'] == id_)].to_list()
    # print(words_sent)
    words_sents.append(words_sent)

    for labels in labels_sent:
        labs.append(labels)
        sents.append(s)
        ids.append(id_)
    for word in words_sent: 
        words.append(word)
    for label_type in label_types:
        lab_types.append(label_type)
    for top_descr in top_descrs:
        top_descripciones.append(top_descr)

df_res_words = pd.DataFrame(zip(ids, sents, words, labs, lab_types, top_descripciones), columns=['file_id', 'sentence', 'words', 'label', 'label_type', 'top_descriptions'])
# df_res_words.columns = ['file_id', 'sentence', 'words', 'label', 'label_type']
# print(df_res_words.head(10))
df_res_words.to_csv('corpus_codiesp/sentences_words-new-rag.tsv', index=False, sep='\t')


icd10_folder = os.path.join('..', 'icd_10_es')
icd10d_file = "ICD10_diagnosticos_2020.tsv"
icd10p_file = "ICD10_procedimientos_2020.tsv"

df_icd10d = pd.read_csv(os.path.join(icd10_folder, icd10d_file), sep='\t', dtype='str')
df_icd10p = pd.read_csv(os.path.join(icd10_folder, icd10p_file), sep='\t', dtype='str')
df_icd10 = pd.concat([df_icd10d, df_icd10p], axis=0)


df_res_words['label'] = df_res_words['label'].str.upper()
df_merged = pd.merge(df_res_words, df_icd10, left_on='label', right_on='codigo', how='left')
df_merged = df_merged[['file_id', 'sentence', 'words', 'label', 'label_type', 'descripcion', 'top_descriptions']]
# df_merged['top_descriptions'] = df_set['top_descriptions'].to_list()
# df_merged['top_codes'] = df_set['top_codes'].to_list()

df_merged.to_csv('corpus_codiesp/sentences_words_icd10-rag.tsv', index=False, sep='\t')

df_empty = df_merged[df_merged['descripcion'].isnull()]
print('EMPTY', len(df_empty))
# print(df_merged.head(10))




