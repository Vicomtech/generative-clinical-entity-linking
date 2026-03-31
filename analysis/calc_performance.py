import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import os 
import numpy as np

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


def calc_mean_recall_at_k(true_labels, predictions, list_k=[1, 8, 16, 32, 64, 128]):
    df = pd.DataFrame()
    scores = []
    for k in list_k: 
        corrects = []
        for label, pred in zip(true_labels, predictions): 
            top_n = pred.split('|')[:k]
            if label in top_n: 
                corrects.append(1)
            else:
                corrects.append(0)
        recall_at_k = round(np.mean(corrects)*100, 2)
        new_row = pd.DataFrame({'K': [k], 'MeanR@K': [recall_at_k]})
        df = pd.concat([df, new_row], ignore_index=True)
        df_transposed = df.T
        new_header = df_transposed.iloc[0]
        df_transposed = df_transposed[1:]
        df_transposed.columns = new_header
    return df_transposed


path = os.getcwd()

starts = 'ICD-10'
code = 'code'

if starts == 'ICD-10':
    code = 'label'

dirs = [d for d in os.listdir(path) 
        if os.path.isdir(os.path.join(path, d)) and d.startswith(starts)]

print(dirs)
filename = 'sentbert_result.tsv'
dfs = []
dfs_k = []
for folder in dirs:
    print(folder)
    if not os.path.exists(os.path.join(folder, filename)): 
        print('No file', filename)
        continue
    df = pd.read_csv(os.path.join(folder, filename), sep='\t', dtype=str) 
    # print(df.head())
    true_labels = df[code].str.upper().to_list()
    predictions = df['top_codes'].str.upper().to_list()

    predicted_labels = []
    for codes in predictions: 
        predicted_labels.append(codes.split('|')[0].upper())

    scores = calc_performance_scores(true_labels, predicted_labels)
    scores['model'] = folder
    scores = scores[['model', 'Acc', 'P', 'R', 'F1']]
    dfs.append(scores)
    print(scores)

    df_k = calc_mean_recall_at_k(true_labels, predictions)
    df_k['model'] = folder
    df_k = df_k[['model', 1, 8, 16, 32, 64, 128]]
    # print(df_k)
    dfs_k.append(df_k)
    print('-----------')

df_res = pd.concat(dfs, ignore_index=True)
df_res.to_csv(starts+'-bert-scores.tsv', sep='\t', index=False)
print(df_res.head())

df_res_k = pd.concat(dfs_k, ignore_index=True)
df_res_k.to_csv(starts+'-bert-meanr_at_k.tsv', sep='\t', index=False)
print(df_res_k.head())

