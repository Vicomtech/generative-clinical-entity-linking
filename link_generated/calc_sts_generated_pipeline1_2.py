import numpy as np
import pandas as pd
import torch
import time
import pickle 
import fnmatch
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from eval.faiss_utils import faiss_search
from eval.transformer_utils import texts2vectors, cls_pooling, get_chunks, flatten, mean_pooling
from sklearn.metrics import f1_score
from eval.evaluation_utils import clean_text, get_icd_code, calculate_code_accuracy


def find_files(directory, pattern):
    filepaths = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            filepaths.append(os.path.join(root, filename))
    return filepaths


def calculate_recall_at_k(df, k=1): 
    # if reference code is in the top k predictions
    top_n = []
    for i, row in df.iterrows():
        predictions = row['top_codes'].split('|')
        reference = row['label'].lower()
        if reference in predictions[:k]:
            top_n.append(1)
        else:
            top_n.append(0)
    return round((np.array(top_n).sum()/len(top_n))*100, 2), top_n

def calculate_accuracy(df): 
    # if predicted description (r@1) is equal to the reference
    corrects = []
    for i, row in df.iterrows():
        predictions = row['top_descriptions'].split('|')    
        prediction = predictions[0].strip()
        prediction = clean_text(prediction)
        # reference  = df_icds[df_icds['codigo'] == label]['descripcion'].values[0]
        reference = row['descripcion'].strip()
        reference = clean_text(reference)

        # print('PRED', prediction)
        # print('REF', reference)

        if prediction == reference:
            corrects.append(1)
        else:
            corrects.append(0)
    return round((np.array(corrects).sum()/len(corrects))*100, 2)


def get_correct_not_accurate(df): 
    df_filtered = df[(df['is_acc'] == 0) & (df['is_recall_correct'] == 1)]
    return df_filtered

def calculate_accuracy_generated(df): 
    # if generated is equal to the reference
    df['descripcion_cl'] = df['descripcion'].apply(clean_text)
    df['pred_cl'] = df['pred'].apply(clean_text)
    accs = []
    for i, row in df.iterrows():
        if row['pred_cl'] == row['descripcion_cl']:
            accs.append(1)
        else:
            accs.append(0)
    return accs

def calculate_embeddings(test_list, output_folder, prefix): 
    chunk_size = 8000
    filename = f"{prefix}_embeddings.pkl"
    output_file = os.path.join(output_folder, filename)

    if os.path.exists(output_file):
        print('Embeddings exist, loading')
        embeddings = pickle.load(open(output_file, "rb"))
    else:
        print(f'Calculating {prefix} embeddings {len(test_list)}')
        test_chunks = get_chunks(test_list, chunk_size)
        print('Chunks', len(test_chunks))
        embeddings_chunks = []
        for i, test in enumerate(test_chunks): 
            print('Chunk', i)
            model_output, attention_mask = texts2vectors(test)
            embeddings0 = cls_pooling(model_output)
            # embeddings0 = mean_pooling(model_output, attention_mask)
            embeddings0 = embeddings0.cpu().detach().numpy()
            embeddings_chunks.append(embeddings0)

        embeddings = flatten(embeddings_chunks)
        with open(output_file, 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return embeddings

cache_folder = os.environ.get('CACHE_FOLDER') or os.environ.get('HF_HOME') or None
if cache_folder:
    os.environ["HF_HOME"] = cache_folder

# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
# os.environ['TOKENIZERS_PARALLELISM']= "false"
print("CUDA available:", torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"

corpus_mapped = 'mapped_corpora'
folder = os.path.join('..', 'corpus')
df_mantra = pd.read_csv(os.path.join(folder, corpus_mapped, 'Spanish_mantra_icd10.tsv'), sep='\t', dtype=str)
df_e3c = pd.read_csv(os.path.join(folder, corpus_mapped, 'Spanish_e3c_icd10_sentence_layer1.tsv'), sep='\t', dtype=str)
df_medterm = pd.read_csv(os.path.join(folder, corpus_mapped, 'medterm_all_icd10_selected.tsv'), sep='\t', dtype=str)

df_mantra = df_mantra[['file_id', 'entity', 'CODE', 'sentence', 'STR']]
df_mantra = df_mantra.reset_index(drop=True)

df_e3c = df_e3c[['file_id', 'entity', 'CODE', 'sentence', 'STR']]
df_e3c = df_e3c.reset_index(drop=True)

df_medterm = df_medterm[['file_id', 'entity', 'CODE', 'sentence', 'STR']]
df_medterm = df_medterm.reset_index(drop=True)

df_all_mapped = pd.concat([df_mantra, df_e3c, df_medterm], ignore_index=True)
print('Total mapped', len(df_all_mapped))

## TRAIN 
corpus = 'codiesp'
part = 'test'
corpus_folder = os.path.join('..', 'corpus')
data_path = os.path.join(corpus_folder, corpus, 'sentences_words')
data_train = os.path.join(data_path, 'train_sentences_words_icd10.tsv')
df_train = pd.read_csv(data_train, sep='\t', dtype='str')
data_dev = os.path.join(data_path, 'dev_sentences_words_icd10.tsv')
df_dev = pd.read_csv(data_dev, sep='\t', dtype='str')

df_all = pd.concat([df_train, df_dev], ignore_index=True)

df_mapped_codiesp = pd.concat([df_all, df_all_mapped], ignore_index=True)
### ICD-10
icd_codes_dir = os.path.join('..', 'icd_10_es')
df_icd_d = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_diagnosticos_2020.tsv'), sep='\t', dtype='str')
df_icd_p = pd.read_csv(os.path.join(icd_codes_dir, 'ICD10_procedimientos_2020.tsv'), sep='\t', dtype='str')
df_icds = pd.concat([df_icd_d, df_icd_p], ignore_index=True)
print('Total ICD-10', len(df_icds))

output_embeddings_folder = os.path.join('..', 'icd_10_es', 'embeddings')

icd10_diag_embeddings = calculate_embeddings(
    df_icd_d['descripcion'].to_list(), 
    output_folder=output_embeddings_folder, prefix='icd10_diag'
    )

icd10_proc_embeddings = calculate_embeddings(
    df_icd_p['descripcion'].to_list(), 
    output_folder=output_embeddings_folder, prefix='icd10_proc'
    )

## DATA
seed = 42
output_folder = 'output_codiesp/Medical-mT5-large-term-sentence-mapped-v2-seed42'
test_filename = f'{part}_predictions.tsv'
filepaths = find_files(output_folder, test_filename)
print(filepaths)

label_types = ['DIAGNOSTICO', 'PROCEDIMIENTO'] 
query_column = 'pred'
accuaracies = []
recalls = []
filepaths_names = []
recalls_unseen = []
accuaracies_unseen = []
icd_accuracies = []
icd_accuracies_unseen = []
for filepath in filepaths:
    print('FILE', filepath)
    filepath_split = filepath.split('/')[-3:-1]
    filepaths_names.append('_'.join(filepath_split))
    dfs = []
    test_df = pd.read_csv(filepath, sep='\t', dtype=str).fillna('')
    predictions = test_df['pred'].to_list()
    true_labels = test_df['label'].to_list()
    
    codes_icd  = get_icd_code(predictions, df_icds)
    codes_acc, codes_accs_col = calculate_code_accuracy(codes_icd, true_labels)
    test_df['is_code_correct'] = codes_accs_col

    icd_accuracies.append(codes_acc)
    print('CODE accuracy', codes_acc)

    if 'words' not in test_df.columns:
        test_df = test_df.rename(columns={'input': 'words'})

    for label_type in label_types:
        df_corpus = test_df[test_df['label_type'] == label_type] 
        df_corpus[query_column] = df_corpus[query_column].str.strip()
        queries = df_corpus[query_column].str.lower().to_list()
        # print('Total QUERIES', len(queries))
        labels = df_corpus.label.str.lower().to_list()

        model_name_with_seed = filepath_split[1]  # e.g., 'mt5-large-term-sentence-mapped-seed456'
        model_embeddings_folder = os.path.join(output_embeddings_folder, model_name_with_seed)
        if not os.path.exists(model_embeddings_folder):
            os.makedirs(model_embeddings_folder)
        
        corpus_embeddings = calculate_embeddings(
            queries, 
            output_folder=model_embeddings_folder,
            prefix=f"{label_type}_{part}"
            )

        if label_type == 'DIAGNOSTICO': 
            descriptions = df_icd_d.descripcion.str.lower().to_list()
            icd_codes = df_icd_d.codigo.str.lower().to_list()
            descriptions_embeddings = icd10_diag_embeddings
        elif label_type == 'PROCEDIMIENTO':
            descriptions = df_icd_p.descripcion.str.lower().to_list()
            icd_codes = df_icd_p.codigo.str.lower().to_list()
            descriptions_embeddings = icd10_proc_embeddings

        d = len(descriptions_embeddings[0])

        D, I = faiss_search(descriptions_embeddings, corpus_embeddings, k=10, d=d)

        result_dicts = []
        for inds, ds, vecs, label in zip(I, D, corpus_embeddings, labels): #inds in ICD-10 corpus
            top_descriptions = '|'.join([descriptions[i] for i in inds])
            top_predictions = '|'.join([icd_codes[i] for i in inds])
            top1_score = float(ds[0])  # cosine similarity score for the top-1 prediction [0, 1]
            result_dicts.append({'top_descriptions': top_descriptions, 'distancies': ds, 'top_codes': top_predictions, 'top1_similarity': top1_score})

        df_res = pd.DataFrame(result_dicts)
        df_corpus = df_corpus.copy()
        df_corpus.loc[:, 'top_descriptions']  = df_res['top_descriptions'].values
        df_corpus.loc[:, 'top_codes'] = df_res['top_codes'].values
        df_corpus.loc[:, 'top1_similarity'] = df_res['top1_similarity'].values
        dfs.append(df_corpus)

    result_df = pd.concat(dfs, axis=0)

    recall_at_k, top_ns = calculate_recall_at_k(result_df, k=1)
    acc_gen = calculate_accuracy_generated(result_df) 

    result_df['is_recall_correct'] = top_ns
    result_df['is_acc'] = acc_gen

    accuracy = calculate_accuracy(result_df)
    recalls.append(recall_at_k)
    accuaracies.append(accuracy)
    # --- Similarity score analysis ---
    correct_mask = result_df['is_recall_correct'] == 1
    all_scores = result_df['top1_similarity'].values
    correct_scores = result_df.loc[correct_mask, 'top1_similarity'].values
    incorrect_scores = result_df.loc[~correct_mask, 'top1_similarity'].values

    def score_stats(scores, label):
        row = {'group': label, 'count': len(scores)}
        if len(scores) > 0:
            row.update({
                'mean':   round(float(np.mean(scores)), 4),
                'std':    round(float(np.std(scores)), 4),
                'min':    round(float(np.min(scores)), 4),
                'max':    round(float(np.max(scores)), 4),
                'median': round(float(np.median(scores)), 4),
            })
            for p in [5, 10, 25, 50, 75, 90, 95]:
                row[f'p{p}'] = round(float(np.percentile(scores, p)), 4)
        return row

    score_rows = [
        score_stats(all_scores, 'all'),
        score_stats(correct_scores, 'correct (recall@1=1)'),
        score_stats(incorrect_scores, 'incorrect (recall@1=0)'),
    ]
    df_scores = pd.DataFrame(score_rows)
    print(f'\n--- Similarity scores ---')
    print(df_scores.to_string(index=False))
    print()

    scores_folder = 'similarity_scores'
    if not os.path.exists(scores_folder):
        os.makedirs(scores_folder)
    scores_filename = os.path.join(scores_folder, f"{'_'.join(filepath_split)}_similarity_scores.tsv")
    df_scores.to_csv(scores_filename, sep='\t', index=False)
    print(f'Similarity scores saved to {scores_filename}')
    print()

    result_df.to_csv(os.path.join('error_analysis', '_'.join(filepath_split)+'.tsv'), sep='\t', index=False)
    print('Recall@1', recall_at_k)
    print('TextAccuracy', accuracy)
    df_not_accurate = get_correct_not_accurate(result_df)

    not_acc_folder = 'error_analysis_not_accurate'
    if not os.path.exists(not_acc_folder):
        os.makedirs(not_acc_folder)

    df_not_accurate.to_csv(os.path.join(not_acc_folder, '_'.join(filepath_split)+'_not_accurate.tsv'), sep='\t', index=False)
   ###UNSEEN DATA
    df_unseen = result_df[~result_df['words'].isin(df_all['words'].values)]
    if 'mapped' in filepath:
        df_unseen = result_df[~(result_df['words'].isin(df_mapped_codiesp['words'].values))]
        
    print('Unseen data: ', len(df_unseen))

    if len(df_unseen) > 0:
        # print(df_unseen)
        # print(df_unseen.columns)
        recall_at_k_unseen, _ = calculate_recall_at_k(df_unseen, k=1)
        recalls_unseen.append(recall_at_k_unseen)
        accuracy_unseen = calculate_accuracy(df_unseen)
        print('TextAccuracy unseen', accuracy_unseen)
        accuaracies_unseen.append(accuracy_unseen)
        print('Recall@1 unseen', recall_at_k_unseen)
        codes_unseen = get_icd_code(df_unseen['pred'].to_list(), df_icds)
        accuracy_unseen_code, _ = calculate_code_accuracy(codes_unseen, df_unseen['label'].to_list())
        print('Code accuracy unseen', accuracy_unseen_code) 
        icd_accuracies_unseen.append(accuracy_unseen_code)
    else:
        recalls_unseen.append(np.nan)
        accuaracies_unseen.append(np.nan)
        icd_accuracies_unseen.append(np.nan)
    print('DONE'+'='*50)
    print()
    
df_table = pd.DataFrame({
    'models': filepaths_names, 
    'CodeAcc': icd_accuracies, 
    'CodeRecall@1': recalls, 
    'TextAccuracy': accuaracies, 
    'CodeAcc_unseen': icd_accuracies_unseen,
    'CodeRecall@1_unseen': recalls_unseen, 
    'TextAccuracy_unseen': accuaracies_unseen}
    )

df_table = df_table.sort_values(by='models')
print(df_table)
df_table.to_csv(f'{output_folder}/{part}_sts_scores.tsv', sep='\t', index=False)
