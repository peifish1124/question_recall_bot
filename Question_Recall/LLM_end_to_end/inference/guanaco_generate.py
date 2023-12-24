import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from  utils import (
    get_prompt,
    get_bnb_config,
)
import json
from tqdm import tqdm
import argparse

max_new_tokens = 640
top_p = 0.9
temperature=0.7

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Taiwan-LLM-7B-v2.0-chat')
    parser.add_argument('--adapter_path', type=str, default='adapter_model')
    parser.add_argument('--input_file_path', type=str, default='data/private_test.json')
    parser.add_argument('--output_file_path', type=str, default='predict.json')
    return parser.parse_args()

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(get_prompt(user_question), return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


args = get_args()
# Base model
model_name_or_path = args.model_name_or_path
# Adapter name on HF hub or local checkpoint path.
adapter_path = args.adapter_path

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# Fixing some of the early LLaMA HF conversion issues.
tokenizer.bos_token_id = 1

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

input_file_path = args.input_file_path
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
output = []
i = 0
for item in tqdm(data):
    if i == 50:
        break
    i += 1
    assistant_response = generate(model, item['instruction'])
    start_index = assistant_response.find("ASSISTANT: ")
    if start_index != -1:
        extracted_text = assistant_response[start_index + len("ASSISTANT: "):]
    output.append({'id': item['id'], 'instruction':item['instruction'] ,'output':extracted_text}) 

result_file_path = args.output_file_path
with open(result_file_path, 'w',  encoding='utf-8') as file:
    json.dump(output, file ,ensure_ascii=False, indent=4)
