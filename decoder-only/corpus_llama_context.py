import pandas as pd 
import os


## DATA
corpus = 'codiesp'
data_path = os.path.join('..', 'corpus', corpus, 'sentences_words')
data_train = os.path.join(data_path, 'train_sentences_words_icd10.tsv')
data_dev = os.path.join(data_path, 'dev_sentences_words_icd10.tsv')
test = os.path.join(data_path, 'test_sentences_words_icd10.tsv')

train_df = pd.read_csv(data_train, sep='\t', dtype=str)
train_df = train_df.sample(frac=1, random_state=42)
train_df.reset_index(drop=True, inplace=True) # necessary for dataloader

eval_df = pd.read_csv(data_dev, sep='\t', dtype=str)    
eval_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(test, sep='\t', dtype=str)    

df = pd.concat([train_df, eval_df])
print('Total', len(df))
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True) # necessary for dataloader

n = len(df) // 10
train_df = df.iloc[n:]
train_df.to_csv('train_sent.tsv', sep='\t', index=False)
eval_df = df.iloc[:n]
eval_df.reset_index(drop=True, inplace=True)

print('Train', len(train_df))
print('Eval', len(eval_df))
print('Source:', train_df['words'].iloc[0])
print('Target', train_df['descripcion'].iloc[0])

def make_intruction_corpus(df): 
    system = "Eres un asistente médico útil, responsable y honesto. Eres experto en la Clasificación Internacional de Enfermedades CIE-10 y puedes generar definiciones de términos médicos."
    instruction = "Genera la definición del siguiente término médico en la Clasificación Internacional de Enfermedades CIE-10. Usa el contexto proporcionado."
    inputs = df['words'].tolist()
    labels = df['label'].tolist()
    outputs = df['descripcion'].tolist()
    entities = df['label_type'].tolist()
    contexts = df['sentence'].tolist()
    inputs_with_entities = [f"Término: {input}; grupo semántico: {entity.lower()}" for input, entity in zip(inputs, entities)]
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

df_res_train = make_intruction_corpus(train_df)
df_res_dev = make_intruction_corpus(eval_df)
df_res_test = make_intruction_corpus(test_df)
print(df_res_train.head())

result_corpus_folder = 'corpus_codiesp'
df_res_test.to_csv(os.path.join(result_corpus_folder, 'test_system_context.tsv'), sep='\t', index=False)
print('Total test:', len(df_res_test))
df_res_dev.to_csv(os.path.join(result_corpus_folder, 'dev_system_context.tsv'), sep='\t', index=False)
print('Total dev:', len(df_res_dev))
df_res_train.to_csv(os.path.join(result_corpus_folder, 'train_system_context.tsv'), sep='\t', index=False)
print('Total train:', len(df_res_train))

# mode_output_templ = 'output_codiesp/Meta-Llama-3.1-8B-llama-template/checkpoint-285/test_predictions.tsv'
# df_templ = pd.read_csv(mode_output_templ, sep='\t', dtype=str)

# model_output_context = 'output_codiesp/Meta-Llama-3.1-8B-system-context/checkpoint-4418/test_predictions.tsv'
# df_context = pd.read_csv(model_output_context, sep='\t', dtype=str)

# df_templ['label'] = df_res_test['label'].tolist()
# df_templ.to_csv('test_predictions_templ.tsv', sep='\t', index=False)
# df_context['label'] = df_res_test['label'].tolist()
# df_context.to_csv('test_predictions_context.tsv', sep='\t', index=False)

