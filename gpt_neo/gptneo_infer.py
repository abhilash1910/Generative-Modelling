from transformers import GPTNeoConfig,AutoTokenizer
from gpt_neo.gpt_neo import GPTNeoModel

config= GPTNeoConfig()
gpt_neo_model= GPTNeoModel(config)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print('Model Logits',outputs.last_hidden_state)
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print('gen_state',gen_text)
