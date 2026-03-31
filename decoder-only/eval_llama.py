import os
import pandas as pd
from eval.evaluation_utils import calculate_all_scores
from eval.transformer_utils import get_model_and_tokenizer, generate_batch
from transformers import GenerationConfig
from datasets import Dataset


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def formatted_user(instruction, input)->str:
    return f"<|im_start|>user\n{instruction}: {input}<|im_end|>\n<|im_start|>assistant"

def prepare_user(data_df):
    data_df["text"] = data_df[["instruction", "input"]].apply(lambda x: formatted_user(x['instruction'], x['input']), axis=1)
    return data_df

def formatted_test(system, inp)->str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant"

def prepare_system_instr_user(data_df):
    data_df['input_context'] = data_df['instruction'] + ' ' + data_df['input_with_entity'] + '; contexto: ' + data_df['context']
    data_df["text"] = data_df[["system", "input_context"]].apply(lambda x: formatted_test(x['system'], x['input_context']), axis=1)
    # data = Dataset.from_pandas(data_df)
    return data_df


corpus = "codiesp"
model_id_base = "meta-llama/Meta-Llama-3.1-8B"

param = "system-instruct-context-mapped-en"
mapped = 'mapped'

output_model= f"./output_{corpus}/{model_id_base.split('/')[-1]}-{param}"
checkpoint = 15479
eval_model_dir = f"{output_model}/checkpoint-{checkpoint}"
print(f"Evaluating model in {eval_model_dir}")
## DATA 
# train_file = f"corpus_{corpus}/train_system_context.tsv"
# dev_file = f"corpus_{corpus}/dev_system_context.tsv"
train_file = f"corpus_{corpus}/train_en.tsv"
dev_file = f"corpus_{corpus}/dev_en.tsv"

df_tr = pd.read_csv(train_file, sep='\t', dtype=str)
df_d = pd.read_csv(dev_file, sep='\t', dtype=str)

corpus_mapped = 'mapped'
# df_train_mapped = pd.read_csv(f"corpus_{corpus_mapped}/train_system_instruction_context.tsv", sep='\t', dtype=str)
# df_dev_mapped = pd.read_csv(f"corpus_{corpus_mapped}/dev_system_instruction_context.tsv", sep='\t', dtype=str)
df_all_mapped = pd.read_csv(f"corpus_{corpus_mapped}/all_en.tsv", sep='\t', dtype=str)

df_train = df_tr
df_dev = df_d

if mapped == 'mapped':
    print('Using mapped data')
    df_train = pd.concat([df_tr, df_all_mapped])
    

print('Train:', len(df_train))
print('Dev:', len(df_dev))
df = pd.concat([df_train, df_dev])

test_df = pd.read_csv(f"corpus_{corpus}/test_en.tsv", sep='\t', dtype=str)
df_base = df_dev
part = 'test'
if part == 'test':
    df_base = test_df
    print('Test:', len(df_base))

dataset_dev = prepare_system_instr_user(df_base)
print('Dataset:', len(dataset_dev))
print(dataset_dev['text'][0])

model, tokenizer = get_model_and_tokenizer(eval_model_dir)

generated_responses = generate_batch(
    dataset_dev['text'].tolist(), 
    tokenizer, 
    model, 
    batch_size=30, 
    max_new_tokens=128
    )
print(f"Generated {len(generated_responses)} responses")

predictions = []
for resp in generated_responses:
    res_formatted = resp.split('<|im_end|>')[0].strip().replace('\n', '')
    predictions.append(res_formatted)

"""## Evaluate the model on the test set"""
descriptions = df_base['output'].to_list()

df_scores, sscores = calculate_all_scores(predictions, descriptions)
df_scores.to_csv(os.path.join(eval_model_dir, part+'_scores.tsv'), sep='\t', index=False)
print('PERFORMANCE')
print(df_scores.head())

dataset_dev['pred'] = predictions
dataset_dev['semscore'] = sscores
dataset_dev.to_csv(os.path.join(eval_model_dir, part+'_predictions.tsv'), sep='\t', index=False)

### UNSEEN DATA
df_unseen = df_base[~df_base['input'].isin(df['input'].values)]
print('Unseen data: ', len(df_unseen))

if len(df_unseen) > 0:
    predictions_lst = df_unseen['pred'].to_list()
    description_list = df_unseen['output'].to_list()

    df_scores, sscores = calculate_all_scores(predictions_lst, description_list)
    df_scores.to_csv(os.path.join(eval_model_dir, part+'_unseen_scores.tsv'), sep='\t', index=False)
    print('UNSEEN PERFORMANCE')
    print(df_scores.head())

    df_seen = df_base[df_base['input'].isin(df['input'].values)]
    predictions_lst = df_seen['pred'].to_list()
    description_list = df_seen['output'].to_list()

    df_scores, sscores = calculate_all_scores(predictions_lst, description_list)
    df_scores.to_csv(os.path.join(eval_model_dir, part+'_seen_scores.tsv'), sep='\t', index=False)
    print('SEEN PERFORMANCE')
    print(df_scores.head())