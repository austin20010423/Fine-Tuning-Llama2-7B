import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import wandb

model = AutoModelForCausalLM.from_pretrained(
    "test_model/epoch0",
    device_map="cuda",
    use_cache=None,
    attn_implementation=None,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

user_prompt = '''What is the sum of 20 + 30?'''
batch = tokenizer(user_prompt, return_tensors="pt")

batch = {k: v.to("cuda") for k, v in batch.items()}
with torch.no_grad():
    outputs = model.generate(
        **batch,
        max_new_tokens=50, #The maximum numbers of tokens to generate
        do_sample=False, #Whether or not to use sampling ; use greedy decoding otherwise.
        top_p=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature=0, # [optional] The value used to modulate the next token probabilities.
        min_length=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache=True, #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_k=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
        output_hidden_states= True, return_dict_in_generate=True,
    )
batch = {k: v.to("cpu") for k, v in batch.items()}


output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print("The complete decode:",output_text)
print("--------------------------------")
print("Just the model's response",output_text[len(user_prompt):])