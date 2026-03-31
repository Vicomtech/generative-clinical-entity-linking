import pandas as pd
import os 

df_antidote = pd.read_csv(os.path.join('antidote', 'antidote_result_correct.tsv'), sep='\t')
print('ANTIDOTE', len(df_antidote))
df_sentbert = pd.read_csv(os.path.join('ICD-10-CodiEsp-SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-add_train', 'sentbert_result_correct.tsv'), sep='\t')
print('SENTBERT', len(df_sentbert))
# Upper bound

corrects_antodote = df_antidote['correct'].tolist()

df_sentbert['correct_antidote'] = corrects_antodote


# Upper bound

def is_correct(row):
    if row['correct'] == 1 or row['correct_antidote'] == 1:
        return 1
    return 0

df_sentbert['correct_upperbound'] = df_sentbert.apply(is_correct, axis=1)
print('UPPERBOUND', len(df_sentbert[df_sentbert['correct_upperbound'] == 1]))

print('Accuracy', len(df_sentbert[df_sentbert['correct_upperbound'] == 1]) / len(df_sentbert))
