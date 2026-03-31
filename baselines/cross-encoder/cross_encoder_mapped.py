import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
import pandas as pd

from datetime import datetime
import os
import gzip
import csv
from zipfile import ZipFile

# Set environment variable for WandB integration
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

import wandb

# Configuration variables
k = 128
corpus_type = 'codiesp_mapped'
model_for_corpus = 'sapbert'
train_batch_size = 256  # Adjusted for 48GB GPU
num_epochs = 20
model_name = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"

# Initialize wandb with config
wandb.init(
    project="MAPPED_cross_encoder", 
    entity=os.environ.get('WANDB_ENTITY', 'elena_zotova_r'),
    name=f'cross_encoder-{corpus_type}-{model_for_corpus}-{k}',
    config={
        "model_name": model_name,
        "k": k,
        "corpus_type": corpus_type,
        "model_for_corpus": model_for_corpus,
        "train_batch_size": train_batch_size,
        "num_epochs": num_epochs,
    }
)


print('CUDA available: ', torch.cuda.is_available())
print('GPU: ', torch.cuda.current_device())

### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

logger.info("Read train dataset")

query = 'source'
result = 'target'
label = 'label'

corpus_codiesp_folder = 'corpus_cross_encoder_codiesp_sapbert_128'
corpus_mapped_folder = 'ICD-10-CodiEsp-SapBERT-UMLS-2020AB-all-lang-from-XLMR-large-icd_only'

df_train_codiesp = pd.read_csv(os.path.join(corpus_codiesp_folder, f"train_cross_encoder_corpus.tsv"), sep='\t').fillna('')
df_dev_codiesp = pd.read_csv(os.path.join(corpus_codiesp_folder, f"dev_cross_encoder_corpus.tsv"), sep='\t').fillna('')

df_train_mapped = pd.read_csv(os.path.join(corpus_mapped_folder, f"train_cross_encoder_corpus.tsv"), sep='\t').fillna('')
df_dev_mapped = pd.read_csv(os.path.join(corpus_mapped_folder, f"dev_cross_encoder_corpus.tsv"), sep='\t').fillna('')

df_train = pd.concat([df_train_codiesp, df_train_mapped], ignore_index=True)
df_dev = pd.concat([df_dev_codiesp, df_dev_mapped], ignore_index=True)

print('Train unique', len(df_train))
print('Dev unique', len(df_dev))

train_samples = []
for _, row in df_train.iterrows():
    #As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
    train_samples.append(InputExample(texts=[row[query], row[result]], label=int(row[label])))
    train_samples.append(InputExample(texts=[row[result], row[query]], label=int(row[label])))

print("Samples in Train {}".format(len(train_samples)))

logger.info("Read dev dataset")
dev_samples = []
dev_pairs = []
dev_labels = []
for _, row in df_dev.iterrows():
    #As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the dev set
    dev_samples.append(InputExample(texts=[row[query], row[result]], label=int(row[label])))
    dev_samples.append(InputExample(texts=[row[result], row[query]], label=int(row[label])))
    dev_pairs.append([row[query], row[result]])
    dev_pairs.append([row[result], row[query]])
    dev_labels.append(int(row[label]))
    dev_labels.append(int(row[label]))

print("Samples in Dev {}".format(len(dev_samples)))

# dev_samples = dev_samples[:1000]
print()

# Calculate training parameters
eval_steps = int(len(train_samples) / train_batch_size)
print("Evaluation steps: {}".format(eval_steps))
model_save_path = f'output/cross_encoder-{corpus_type}-{model_for_corpus}-{k}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}' 

#We use distilroberta-base with a single label, i.e., it will output a value between 0 and 1 indicating the similarity of the two questions
label2int = {"false": 0, "true": 1}

model = CrossEncoder(model_name, num_labels=1)
# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=dev_pairs,
    labels=dev_labels,
    name=f'{corpus_type}-{k}-{model_for_corpus}-performance',
    write_csv=True
)
# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.05) #10% of train data for warm-up
print("Warmup-steps: {}".format(warmup_steps))

# Custom callback to log metrics to WandB
def wandb_callback(score, epoch, steps):
    """Log evaluation metrics to WandB"""
    wandb.log({
        "eval/score": score,
        "epoch": epoch,
        "step": steps
    })

# Train the model with wandb logging
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=eval_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path, 
          save_best_model=True,
          callback=wandb_callback)
