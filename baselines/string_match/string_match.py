import pandas as pd 
import os


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

corpus_folder = 'corpus/codiesp'
test_file = 'test_sentence_description.tsv'
df_test = pd.read_csv(os.path.join(corpus_folder, test_file), sep='\t') 
df_test['words'] = df_test['words'].str.lower()
df_test['words'] = df_test['words'].apply(lambda x: clean_text(x))
df_test = df_test[["file_id", "label_type", "label", "words", "offset"]]

icd_folder = 'icd_10_es'
icd10_d_file = "ICD10_diagnosticos_2020.tsv" 
df_icd10_d = pd.read_csv(os.path.join(icd_folder, icd10_d_file), sep='\t')

icd10_p_file = "ICD10_procedimientos_2020.tsv"
df_icd10_p = pd.read_csv(os.path.join(icd_folder, icd10_p_file), sep='\t')

df_icd10 = pd.concat([df_icd10_d, df_icd10_p], axis=0)	
df_icd10['descripcion'] = df_icd10['descripcion'].str.lower()
df_icd10 = df_icd10[["codigo", "descripcion"]]
df_icd10['descripcion'] = df_icd10['descripcion'].apply(lambda x: clean_text(x))

df_merged = pd.merge(df_test, df_icd10, left_on='words', right_on='descripcion', how='left')
# print(df_merged.head())

df_matched = df_merged[df_merged['descripcion'].notnull()]
print("Matched: ", len(df_matched))
df_matched.to_csv('matched.tsv', sep='\t', index=False)

accuracy = len(df_matched) / len(df_test)
print("Accuracy: ", accuracy*100)

def code_accuracy(df): 
    accs = []
    for i, row in df.iterrows(): 
        if row['label'] is not None: 
            if row['label'] == row['codigo']: 
                accs.append(1)
            else: 
                accs.append(0)
    df['acc'] = accs
    return df, sum(accs) / len(accs)

df, code_accuracy = code_accuracy(df_merged)

print("Code accuracy: ", code_accuracy*100)

df_correct = df[(df['acc'] != 1) & (df['descripcion'].notnull())]
df_correct.to_csv('correct.tsv', sep='\t', index=False)

print(df_correct.head())
