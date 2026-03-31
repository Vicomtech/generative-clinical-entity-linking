import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('DEVICE', device)


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

# folder = 'ICD-10-CodiEsp-SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-add_train'
# folder = "/DATA/ezotova_data/ICD-10_CodiEsp/baseline_bm25/ICD-10-CodiEsp-BM25-icd_only-nostop"
folder = "ICD-10-CodiEsp-BioLORD-2023-M-icd_only"
# folder = "/DATA/ezotova_data/ICD-10_CodiEsp/embeddings_fasstext/ICD-10-CodiEsp-clinic-icd_only-stop"
# folder = "/DATA/ezotova_data/ICD-10_CodiEsp/baseline_bm25/ICD-10-CodiEsp-BM25-icd_only-stop"
# folder = "SNOMED-MedProcNER-SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-snomed_only"
# df_test = pd.read_csv(f'{folder}/sentbert_result.tsv', sep='\t').fillna('')
df_test = pd.read_csv(f'{folder}/sentbert_result.tsv', sep='\t').fillna('')
# df_test = df_test[:10]

span = 'words'
label = "label"
k = 128

def get_top_k(lst): 
    top_k = lst[:k]
    return top_k

df_test['top_descriptions'] = df_test['top_descriptions'].str.split('|')
df_test['top_codes'] = df_test['top_codes'].str.split('|')
df_test['top_descriptions'] = df_test['top_descriptions'].apply(get_top_k)
df_test['top_codes'] = df_test['top_codes'].apply(get_top_k)

descriptions = df_test['top_descriptions'].tolist()
print(len(descriptions[0]))
queries = df_test[span].tolist()
codes = df_test['top_codes'].tolist()
true_codes = df_test[label].str.lower().tolist()

# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# cross_encoder_name = "/DATA/ezotova_data/ICD-10_CodiEsp/cross_encoder/output_all/training_icd10_cross_encoder-2021-11-11_11-46-23"
# cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
cross_encoder_name = "/DATA/ezotova_data/ICD-10_CodiEsp/cross-encoders/output/codiesp_cross_encoder-codiesp-128-2024-07-17_10-33-22"
# cross_encoder_name = "/DATA/ezotova_data/ICD-10_CodiEsp/cross-encoders/output/codiesp_cross_encoder-medprocner-128-2024-07-15_13-25-13"
# cross_encoder_name = "/DATA/ezotova_data/ICD-10_CodiEsp/cross-encoders/output/codiesp_cross_encoder-64-2024-07-15_12-52-06" # fasttext
# cross_encoder_name = "/DATA/ezotova_data/ICD-10_CodiEsp/cross-encoders/output/codiesp_cross_encoder-codiesp-128-2024-07-17_14-58-35" 
cross_encoder = CrossEncoder(cross_encoder_name)

def get_cross_inp(query, descriptions):
    return [(query, d) for d in descriptions]

predictions = []
scores = []
for q, d, p, c in tqdm(zip(queries, descriptions, codes, true_codes)): 
    cross_scores = cross_encoder.predict(get_cross_inp(q, d))
    cross_scores_tensor = torch.tensor(cross_scores)  
    top_idx = torch.argmax(cross_scores_tensor)  
    cross_score = cross_scores[top_idx]
    scores.append(cross_score)

    # print(f'{cross_scores[top_idx]:.2f} - {q} - True: {c} - Bi: {p[0]} - Cross: {p[top_idx]} - {d[top_idx]}')
    predictions.append(p[top_idx])

df_test['cross-encoder'] = predictions
df_test['cross-encoder-score'] = scores
df_test.to_csv(f'{folder}/sentbert_result_{cross_encoder_name.split("/")[-1]}.tsv', sep='\t', index=False)

df_performance = calc_performance_scores(true_codes, predictions)
df_performance.to_csv(f'{folder}/sentbert_{cross_encoder_name.split("/")[-1]}_scores.tsv', sep='\t', index=False)
print(df_performance.head())
