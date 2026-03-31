"""
eval/transformer_utils.py
Utility functions for encoding text with transformer models and running
batched generation.  This file extends utils/transformer_utils.py with
higher-level helpers required by the evaluation scripts.
"""

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
import os
import gc
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from dotenv import load_dotenv
load_dotenv()

cache_folder = os.environ.get("CACHE_FOLDER") or os.environ.get("HF_HOME") or None
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('DEVICE', device)


# ── Encoding helpers (shared with utils/transformer_utils.py) ─────────────────

def texts2vectors(text_list, model_name):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_folder)
    model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_folder).to(device)
    encoded_input = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return model_output, encoded_input['attention_mask']


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output):
    return model_output[0][:, 0]


def get_chunks(ids, n):
    return [ids[i:i + n] for i in range(0, len(ids), n)]


def flatten(chunks):
    return [item for sublist in chunks for item in sublist]


# ── Seq2Seq batched generation (used by t5_multi_eval_context_mapped_seeds.py) ─

def generate_in_batches(texts, batch_size, tokenizer, model, generation_config=None, device=device):
    """Generate text predictions in batches using a Seq2Seq model.

    Args:
        texts: list of input strings
        batch_size: number of samples per batch
        tokenizer: HuggingFace tokenizer
        model: HuggingFace seq2seq model (already on the correct device)
        generation_config: optional GenerationConfig; uses model defaults if None
        device: torch device string

    Returns:
        list of decoded prediction strings
    """
    model.eval()
    all_predictions = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Generating", unit="batch"):
        batch_texts = texts[i: i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            kwargs = {}
            if generation_config is not None:
                kwargs["generation_config"] = generation_config
            output_ids = model.generate(**inputs, **kwargs)

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_predictions.extend(decoded)

    return all_predictions


# ── Causal LM helpers (used by eval_llama.py) ─────────────────────────────────

def get_model_and_tokenizer(model_id):
    """Load a fine-tuned causal LM (possibly LoRA/PEFT) with 4-bit quantization.

    Args:
        model_id: local path or HuggingFace model ID

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        add_bos=True,
        add_eos=True,
        padding_side="right",
        cache_dir=cache_folder,
    )
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Try loading as a PEFT/LoRA merged model first
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_folder,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_folder,
            quantization_config=bnb_config,
            device_map="auto",
        )

    model.config.use_cache = False
    return model, tokenizer


def generate_batch(texts, tokenizer, model, batch_size=16, max_new_tokens=128):
    """Run batched causal generation and return only the newly generated tokens.

    Args:
        texts: list of prompt strings
        tokenizer: HuggingFace tokenizer
        model: causal LM model
        batch_size: samples per batch
        max_new_tokens: maximum tokens to generate per sample

    Returns:
        list of generated (continuation-only) strings
    """
    model.eval()
    all_outputs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Strip the prompt tokens so we only return newly generated text
        new_tokens = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_outputs.extend(decoded)

    return all_outputs
