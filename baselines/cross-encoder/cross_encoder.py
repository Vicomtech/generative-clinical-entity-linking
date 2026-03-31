import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
import pandas as pd

from datetime import datetime
import os
import gzip
import csv
from zipfile import ZipFile

import wandb
wandb.init(project="MEDPROCNER_cross_encoder", entity=os.environ.get('WANDB_ENTITY', 'elena_zotova_r'))

# environment
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', '')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
gpu_ids = ["4"]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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

k = 128
model_for_corpus = "sapbert"
corpus_type = "medprocner"
corpus_folder = f'corpus_cross_encoder_{corpus_type}_{model_for_corpus}_{k}' # corpus_cross_encoder_medporcner_128

print(corpus_folder)
print(corpus_type)
print(model_for_corpus)

train_file = f"train_cross_encoder_corpus.tsv"
dev_file = f"dev_cross_encoder_corpus.tsv"
train_samples = []
with open(os.path.join(corpus_folder, train_file), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        #As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
        train_samples.append(InputExample(texts=[row[query], row[result]], label=int(row[label])))
        train_samples.append(InputExample(texts=[row[result], row[query]], label=int(row[label])))

print("Samples in Train {}".format(len(train_samples)))

logger.info("Read dev dataset")
dev_samples = []
with open(os.path.join(corpus_folder, dev_file), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_samples.append(InputExample(texts=[row[query], row[result]], label=int(row[label])))

# dev_samples = dev_samples[:1000]
print("Samples in Dev {}".format(len(dev_samples)))
print()

#Configuration
train_batch_size = 72
num_epochs = 20
eval_steps = len(train_samples) / train_batch_size
print("Evaluation steps: {}".format(eval_steps))
model_save_path = f'output/cross_encoder-{corpus_type}-{model_for_corpus}-{k}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}' 

#We use distilroberta-base with a single label, i.e., it will output a value between 0 and 1 indicating the similarity of the two questions
label2int = {"false": 0, "true": 1}

model_name = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
model = CrossEncoder(model_name, num_labels=1)
# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name=f'{corpus_type}-{k}-{model_for_corpus}-performance', write_csv=True)
# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.05) #10% of train data for warm-up
print("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=eval_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path, 
          save_best_model=True)
