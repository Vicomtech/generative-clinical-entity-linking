from transformers import AutoConfig, MT5ForConditionalGeneration, AutoTokenizer

model_name = "ezotova/medical-mt5-clinical-el-spanish"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# The Hub config.json says tie_word_embeddings=True (mT5 default), but the
# model was fine-tuned with broken weight-tying (model_init called .contiguous()
# on all params), so lm_head.weight is saved as an independent tensor.
config = AutoConfig.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name, config=config)

# Example: entity mention from a clinical note
def make_prompt(term, sentence):
    return f"Genera una definición para el término: {term} - en la frase: {sentence}"

term = "TC abdominal"
sentence = "La TC abdominal es normal."

prompt = make_prompt(term, sentence)
print("PROMPT", prompt)

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    num_beams=5,
    early_stopping=True
)

prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(prediction)