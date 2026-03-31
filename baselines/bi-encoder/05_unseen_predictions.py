import pandas as pd

import os

folder = 'ICD-10-CodiEsp-SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-add_train'
df = pd.read_csv(f'{folder}/sentbert_result.tsv', sep='\t').fillna('')

def is_correct(row):
    if row['label'].lower() == row['top_codes'].split('|')[0]:
        return 1
    return 0


def get_code_predictions(row):
    return row['top_codes'].split('|')[0]

def get_description_predictions(row):
    return row['top_descriptions'].split('|')[0]

df['pred_code'] = df.apply(get_code_predictions, axis=1)
df['pred_description'] = df.apply(get_description_predictions, axis=1)
df['correct'] = df.apply(is_correct, axis=1)
df = df[['words', 'label', 'descripcion', 'correct', 'pred_code', 'pred_description']]
print(df['correct'].sum())
print(len(df))
df.to_csv(f'{folder}/sentbert_result_correct.tsv', sep='\t', index=False)

train_dir = os.path.join('..', '..', 'corpus', 'codiesp')
df_train = pd.read_csv(os.path.join(train_dir, 'train_sentences_words_icd10.tsv'), sep='\t').fillna('')
df_train = df_train.drop_duplicates(subset=['words'])
df_train['words'] = df_train['words'].str.lower()
df_train = df_train[["label", "words", "block_label", "label_type"]]


antidote_folder = 'antidote'
df_antidote = pd.read_csv(f'{antidote_folder}/test_predictions.tsv', sep='\t').fillna('')

df_antidote['words'] = df_antidote['words'].str.lower()
df_unseen = df_antidote[~df_antidote['words'].isin(df_train['words'].values)]

print('UNSEEN', len(df_unseen))

def get_descr_predictions(row):
    if row['descripcion'].lower() == row['pred'].lower():
        return 1
    return 0

df_unseen['correct'] = df_unseen.apply(get_descr_predictions, axis=1)
print(df_unseen.head())
print(len(df))
df_unseen.to_csv(f'{antidote_folder}/antidote_result_correct.tsv', sep='\t', index=False)