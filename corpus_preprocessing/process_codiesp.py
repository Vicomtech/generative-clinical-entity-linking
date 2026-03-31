# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:50:40 2020

@author: ezotova

add ICD-10 code descriptions and class descriptions to corpus terms 

"""

import pandas as pd
import os
import glob
import re

import nltk


def text_preprocessing(string):
	string = string.replace('\n', '')
	string = string.replace('.', '. ')
	string = string.replace(',', ', ')
	string = string.replace(';', '; ')
	string = string.replace(':', ': ')
	string = re.sub(r'\s{2,}', ' ', string)
	return string


def parse_start_end(offset):
	offset_split = str(offset).split()
	return int(offset_split[0]), int(offset_split[-1])


def build_sentence_spans(part_texts_dir):
	spans_by_file = {}
	for file_path in glob.glob(os.path.join(part_texts_dir, '*.txt')):
		file_id = os.path.splitext(os.path.basename(file_path))[0]
		with open(file_path, 'r', encoding='utf-8') as f:
			text = text_preprocessing(f.read())

		spans = []
		cursor = 0
		for sentence in nltk.sent_tokenize(text):
			start = text.find(sentence, cursor)
			if start == -1:
				continue
			end = start + len(sentence)
			spans.append((start, end, sentence))
			cursor = end

		spans_by_file[file_id] = spans

	return spans_by_file


def attach_entity_sentence(df_labels, part_texts_dir):
	spans_by_file = build_sentence_spans(part_texts_dir)
	sent_values = []

	for _, row in df_labels.iterrows():
		start, _ = parse_start_end(row['offset'])
		sentence = ''
		for sent_start, sent_end, sent_text in spans_by_file.get(row['file_id'], []):
			if sent_start <= start <= sent_end:
				sentence = sent_text
				break
		sent_values.append(sentence)

	df_labels = df_labels.copy()
	df_labels['sentence'] = sent_values
	return df_labels


    
data_dir = os.path.join("corpus", "codiesp")
train_dir = os.path.join(data_dir, 'train')
train_texts_dir = os.path.join(train_dir, 'text_files')
test_dir = os.path.join(data_dir, 'test')
dev_dir = os.path.join(data_dir, 'dev')

colnames = ['file_id', 'label_type', 'label', 'words', 'offset']
df_labels_test = pd.read_csv(os.path.join(test_dir, 'testX.tsv'), dtype={'label': 'str'}, sep='\t', names=colnames)
df_labels_test['label'] = df_labels_test['label'].str.upper()
df_labels_test = attach_entity_sentence(df_labels_test, os.path.join(test_dir, 'text_files'))

df_labels_train = pd.read_csv(os.path.join(train_dir, 'trainX.tsv'), dtype={'label': 'str'}, sep='\t', names=colnames)
df_labels_train['label'] = df_labels_train['label'].str.upper()
df_labels_train = attach_entity_sentence(df_labels_train, os.path.join(train_dir, 'text_files'))
df_labels_dev = pd.read_csv(os.path.join(dev_dir, 'devX.tsv'), dtype={'label': 'str'}, sep='\t', names=colnames)
df_labels_dev['label'] = df_labels_dev['label'].str.upper()
df_labels_dev = attach_entity_sentence(df_labels_dev, os.path.join(dev_dir, 'text_files'))

#### ICD-10
icd_codes_dir = "icd_10_es"

df_p = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_procedimientos_2020.tsv'), sep='\t', dtype='str', encoding='utf-8')
df_p['codigo'] = df_p['codigo']
df_d = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_diagnosticos_2020.tsv'), sep='\t', dtype='str', encoding='utf-8') 
df_d['codigo'] = df_d['codigo']

frames = [df_p, df_d]
df_icds = pd.concat(frames)
df_icds = df_icds[['codigo', 'descripcion']]

df_icds['codigo'] = df_icds['codigo']
df_icds['descripcion'] = df_icds['descripcion']

# file_id	sentence	words	label	label_type	id	codigo	descripcion	block_label

df_train = pd.merge(df_labels_train, df_icds, left_on='label', right_on='codigo', how='left')
df_train = df_train.fillna('') 
df_train = df_train[['file_id', 'sentence', 'words', 'offset', 'label', 'label_type',  'descripcion', ]]
print(len(df_train))
df_train.to_csv(os.path.join(data_dir, 'train_sentence_description.tsv'), sep='\t', index=False)

df_test = pd.merge(df_labels_test, df_icds, left_on='label', right_on='codigo', how='left')
df_test = df_test.fillna('')
df_test = df_test[['file_id', 'sentence', 'words', 'offset', 'label', 'label_type', 'descripcion']]
print(len(df_test))
df_test.to_csv(os.path.join(data_dir, 'test_sentence_description.tsv'), sep='\t', index=False)

df_dev = pd.merge(df_labels_dev, df_icds, left_on='label', right_on='codigo', how='left')
df_dev = df_dev.fillna('')
df_dev = df_dev[['file_id', 'sentence', 'words', 'offset', 'label', 'label_type', 'descripcion']]
print(len(df_dev))
df_dev.to_csv(os.path.join(data_dir, 'dev_sentence_description.tsv'), sep='\t', index=False)

