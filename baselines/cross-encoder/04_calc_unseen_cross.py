import pandas as pd 
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import os 

def calc_performance_scores(true_labels, preds):
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='macro', zero_division=0)
    recall = recall_score(true_labels, preds, average='macro', zero_division=0)
    f1 = f1_score(true_labels, preds, average='macro', zero_division=0)

    accuracy = round(accuracy*100, 2)
    precision = round(precision*100, 2)
    recall = round(recall*100, 2)
    f1 = round(f1*100, 2)
    
    scores_df = pd.DataFrame({
        "Acc": [accuracy],
        "P": [precision],
        "R": [recall],
        "F1": [f1]
    })
    return scores_df

test_file = '/DATA/ezotova_data/ICD-10_CodiEsp/embeddings_sentbert/ICD-10-CodiEsp-BioLORD-2023-term_only/sentbert_result_h.tsv'

df_test = pd.read_csv(test_file, sep='\t', dtype=str)

train_folder = '/DATA/ezotova_data/ICD-10_CodiEsp/cross-encoders/corpus_cross_encoder_codiesp_sapbert_128'
train_file = 'train_cross_encoder_corpus.tsv'
dev_file = 'dev_cross_encoder_corpus.tsv'

df_train = pd.read_csv(os.path.join(train_folder, train_file), sep='\t', dtype=str)
df_dev = pd.read_csv(os.path.join(train_folder, dev_file), sep='\t', dtype=str)

df_all = pd.concat([df_train, df_dev], ignore_index=True)
df_unseen = df_test[~df_test['words'].str.lower().isin(df_all['source'].str.lower())]

print(len(df_unseen))

true_labels = df_unseen['label'].str.lower().tolist()
preds = df_unseen['cross-encoder'].str.lower().tolist()

scores_codes = calc_performance_scores(true_labels, preds)
print(scores_codes)

df_icd10_diag = pd.read_csv('/DATA/ezotova_data/ICD-10_CodiEsp/data/icd/ICD10_diagnosticos_block_2020.tsv', sep='\t', dtype=str)
df_icd10_proc = pd.read_csv('/DATA/ezotova_data/ICD-10_CodiEsp/data/icd/ICD10_procedimientos_block_2020.tsv', sep='\t', dtype=str)

df_icd = pd.concat([df_icd10_diag, df_icd10_proc], ignore_index=True)

true_textes = df_unseen['descripcion'].str.lower().tolist()
cross_encoder_codes = df_unseen['cross-encoder'].tolist()

pred_definitions = []
for code in cross_encoder_codes:
    code = code.upper()
    df_code = df_icd[df_icd['codigo'] == code]
    definition = df_code['descripcion'].values[0]
    definition = definition.lower()
    pred_definitions.append(definition)



scores_strings = calc_performance_scores(true_textes, pred_definitions)
print(scores_strings)
