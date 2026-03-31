import pandas as pd 
import os


def get_label_type(row): 
    if isinstance(row['SAB'], float):
        print(row)
        
    if 'PCS' in row['SAB']:
        return 'procedimiento'
    else: 
        return 'diagnostico'

## DATA
corpus_mapped = 'mapped_corpora'
mapped_folder = os.path.join('..', 'corpus', corpus_mapped)
df_mantra = pd.read_csv(os.path.join(mapped_folder, 'Spanish_mantra_icd10.tsv'), sep='\t', dtype=str)
df_e3c = pd.read_csv(os.path.join(mapped_folder, 'Spanish_e3c_icd10_sentence_layer1.tsv'), sep='\t', dtype=str)
df_medterm = pd.read_csv(os.path.join(mapped_folder, 'medterm_all_icd10_selected.tsv'), sep='\t', dtype=str)

df_mantra = df_mantra[['file_id', 'entity', 'CODE', 'sentence', 'STR', 'SAB']]
df_mantra = df_mantra.reset_index(drop=True)
df_mantra['corpus'] = 'mantra'
print('Mantra:', len(df_mantra))

df_e3c = df_e3c[['file_id', 'entity', 'CODE', 'sentence', 'STR', 'SAB']]
df_e3c = df_e3c.reset_index(drop=True)
df_e3c['corpus'] = 'e3c'
print('E3C:', len(df_e3c))

df_medterm = df_medterm[['file_id', 'entity', 'CODE', 'sentence', 'STR', 'SAB']]
df_medterm = df_medterm.reset_index(drop=True)
df_medterm['corpus'] = 'medterm'
print('Medterm:', len(df_medterm))

df = pd.concat([df_mantra, df_e3c, df_medterm]).reset_index(drop=True)

df['label_type'] = df.apply(get_label_type, axis=1)

print('Total mapped', len(df))

df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True) # necessary for dataloader

n = len(df) // 10
train_df = df.iloc[n:]
eval_df = df.iloc[:n]
eval_df.reset_index(drop=True, inplace=True)

print('Train', len(train_df))
print('Eval', len(eval_df))
print('Source:', train_df['entity'].iloc[0])
print('Target', train_df['STR'].iloc[0])

def make_intruction_corpus(df): 
    system = "You are a helpful, responsible and honest medical assistant. You are an expert in the International Classification of Diseases ICD-10 and can generate definitions of medical terms in Spanish. "
    instruction = "Generate a clear, concise definition of the following medical term in the International Classification of Diseases ICD-10 in Spanish. Use the context provided. "
    inputs = df['entity'].tolist()
    labels = df['CODE'].tolist()
    outputs = df['STR'].tolist()
    entities = df['label_type'].tolist()
    contexts = df['sentence'].tolist()
    inputs_with_entities = [f"Term: {input}; semantic group: {entity.lower()}" for input, entity in zip(inputs, entities)]
    instructions = [instruction for _ in range(len(df))]
    system_prompts = [system for _ in range(len(df))]
    df_result = pd.DataFrame({
        'system': system_prompts,
        'instruction': instructions, 
        'input': inputs, 
        'output': outputs, 
        'label_type': entities, 
        'input_with_entity': inputs_with_entities,
        'context': contexts, 
        'label': labels
        })
    return df_result

# def make_intruction_corpus(df): 
#     instruction = "Eres un asistente médico útil, responsable y honesto. Genera la definición del siguiente término médico en la Clasificación Internacional de Enfermedades CIE-10"
#     inputs = df['entity'].tolist()
#     labels = df['CODE'].tolist()
#     outputs = df['STR'].tolist()
#     entities = df['label_type'].tolist()
#     contexts = df['sentence'].tolist()
#     inputs_with_entities = [f"término: {input}; grupo semántico: {entity.lower()}" for input, entity in zip(inputs, entities)]
#     instructions = [instruction for _ in range(len(df))]
#     df_result = pd.DataFrame({
#         'instruction': instructions, 
#         'input': inputs, 
#         'output': outputs, 
#         'entity_type': entities, 
#         'input_with_entity': inputs_with_entities,
#         'context': contexts, 
#         'label': labels
#         })
#     return df_result

df_res_train = make_intruction_corpus(train_df)
df_res_dev = make_intruction_corpus(eval_df)
df_res = make_intruction_corpus(df)

# df_res_test = make_intruction_corpus(test_df)
print(df_res_train.head())

result_corpus_folder = 'corpus_mapped_en'
if not os.path.exists(result_corpus_folder):
    os.makedirs(result_corpus_folder)
df_res_dev.to_csv(os.path.join(result_corpus_folder, 'dev_en.tsv'), sep='\t', index=False)
print('Total dev:', len(df_res_dev))
df_res_train.to_csv(os.path.join(result_corpus_folder, 'train_en.tsv'), sep='\t', index=False)
print('Total train:', len(df_res_train))
df_res.to_csv(os.path.join(result_corpus_folder, 'all_en.tsv'), sep='\t', index=False)

print('Total:', len(df_res))