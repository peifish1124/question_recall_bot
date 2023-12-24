import argparse
import json
import math
import os
import random

import nltk
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from filelock import FileLock
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from transformers.utils import is_offline_mode
from copy import deepcopy

import argparse

def parse_args():
    # parse --model_name_or_path
    parser = argparse.ArgumentParser(description='This is the code for options generation.')
    parser.add_argument('--model_name_or_path', type=str, default='google/mt5-base', help='model name or path')
    args = parser.parse_args()
    return args

def main(question, answer, model_name_or_path):

    config_name = f"{model_name_or_path}/config.json"
    tokenizer_name = model_name_or_path


    sentences = [
        {
            "question": question,
            "answer": answer
        }
    ]

    # You should update this to your particular problem to have better documentation of `model_type`
    MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
    MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)
            
        
    def parse_args(myStr=None):
        
        if myStr:
            myStr = myStr.split()
        parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
        parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="The name of the dataset to use (via the datasets library).",
        )
        parser.add_argument(
            "--dataset_config_name",
            type=str,
            default=None,
            help="The configuration name of the dataset to use (via the datasets library).",
        )
        parser.add_argument(
            "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
        )
        parser.add_argument(
            "--ignore_pad_token_for_loss",
            type=bool,
            default=True,
            help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
        )
        parser.add_argument(
            "--max_source_length",
            type=int,
            default=1024,
            help=(
                "The maximum total input sequence length after "
                "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument(
            "--source_prefix",
            type=str,
            default=None,
            help="A prefix to add before every source text (useful for T5 models).",
        )
        parser.add_argument(
            "--preprocessing_num_workers",
            type=int,
            default=None,
            help="The number of processes to use for the preprocessing.",
        )
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument(
            "--max_target_length",
            type=int,
            default=128,
            help=(
                "The maximum total sequence length for target text after "
                "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
                "during ``evaluate`` and ``predict``."
            ),
        )
        parser.add_argument(
            "--val_max_target_length",
            type=int,
            default=None,
            help=(
                "The maximum total sequence length for validation "
                "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
                "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
                "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
            ),
        )
        parser.add_argument(
            "--num_beams",
            type=int,
            default=None,
            help=(
                "Number of beams to use for evaluation. This argument will be "
                "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
            ),
        )
        parser.add_argument(
            "--pad_to_max_length",
            action="store_true",
            help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
        )
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            required=False,
        )
        parser.add_argument(
            "--config_name",
            type=str,
            default=None,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--text_column",
            type=str,
            default=None,
            help="The name of the column in the datasets containing the full texts (for summarization).",
        )
        parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
        )
        parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=8,
            help="Batch size (per device) for the evaluation dataloader.",
        )
        parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
        parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        parser.add_argument(
            "--model_type",
            type=str,
            default=None,
            help="Model type to use if training from scratch.",
            choices=MODEL_TYPES,
        )
        parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
        parser.add_argument(
            "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
        )
        parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
        parser.add_argument(
            "--trust_remote_code",
            type=bool,
            default=False,
            help=(
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            ),
        )
        parser.add_argument(
            "--checkpointing_steps",
            type=str,
            default=None,
            help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="If the training should continue from a checkpoint folder.",
        )
        parser.add_argument(
            "--with_tracking",
            action="store_true",
            help="Whether to enable experiment trackers for logging.",
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="all",
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
                "Only applicable when `--with_tracking` is passed."
            ),
        )

        parser.add_argument(
            "--inference_file",
            type=str,
            default=None,
            help="The name of the inference file to be generated.",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
        )
        if myStr:
            args = parser.parse_args(myStr)
        else:
            args = parser.parse_args()
        return args

    import json

    '''
    Sentences:
    [
        {
            "question": "What is the capital of Taiwan?",
            "answer": "Taipei"
        }, ...
    ]

    Write the sentence to a jsonl file with the following format

    {
    "id":0.
    "split":"train"
    "maintext":f"{question} <extra_id_1> {answer}}"
    }
    '''

    # Write the list of JSON objects to a JSON file

    temp_eval_path = 'temp_eval_erjig3.jsonl'

    with open(temp_eval_path, 'w', encoding = 'utf-8') as outfile:
        for i, entry in enumerate(sentences):
            json.dump({
                "id": i,
                "split": "train",
                "maintext": f"{entry['question']} <extra_id_0> {entry['answer']}"
            }, outfile, ensure_ascii=False)

            outfile.write('\n')


    infernce_file = 'output_ferjgjlfe.jsonl'
    

    args = parse_args(f'''
    --validation_file {temp_eval_path}
    --preprocessing_num_workers 8 
    --max_target_length 64 
    --val_max_target_length 64 
    --num_beams 1 
    --model_name_or_path {model_name_or_path}
    --config_name {config_name}
    --tokenizer_name {tokenizer_name}
    --text_column maintext 
    --per_device_eval_batch_size {min(32, len(sentences))} 
    --output_dir ./working/ 
    --inference_file {infernce_file}
    ''')
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(**accelerator_log_kwargs)

    
    import json
    import jsonlines
    
    json_objects = []
    new_eval_path = f"temp_agjklrwj43.json"
    # Read the JSONlines file and convert it to a list of JSON objects
    with open(args.validation_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            json_objects.append(json.loads(line))

    # Write the list of JSON objects to a JSON file
    with open(new_eval_path, 'w', encoding = 'utf-8') as outfile:
        json.dump(
            {'data': json_objects}, outfile, indent=2, ensure_ascii = False)
    
    del json_objects
        
    args.validation_file = new_eval_path
    
    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field = 'data')

    config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        trust_remote_code=args.trust_remote_code,
    )
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["validation"].column_names

    # Get the column names for input/target.
    dataset_columns = ('title', 'maintext')
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )


    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        #print('id', examples['id'])
        #print('inputs_id', model_inputs['input_ids'])
        model_inputs['id'] = [[int(j) for i in range(256)] for j in examples['id']]
        return model_inputs
    
    if args.debug:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(16))

    with accelerator.main_process_first():
        max_target_length = args.val_max_target_length
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        return preds

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

        
    loss_logs = []
        
    model.eval()

    gen_kwargs = {
        "max_length": args.val_max_target_length,
        "num_beams": args.num_beams,
        "repetition_penalty": 3.0
    }

    eval_output = []
    

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, grassID = accelerator.gather_for_metrics((generated_tokens, batch['id']))
            generated_tokens = generated_tokens.cpu().numpy()
            grassID = grassID.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            decoded_preds = postprocess_text(decoded_preds)

            decoded_preds = [title.lstrip("<extra_id_0>") for title in decoded_preds]
            
            id_list = [i[0] for i in grassID]
            
            for grass in range(len(decoded_preds)):
                eval_output.append({
                    "id": str(id_list[grass]),
                    "title": decoded_preds[grass]
                })
    
        



    os.remove(temp_eval_path)
    os.remove(new_eval_path)
    
    for i, entry in enumerate(sentences):
        entry['output'] = eval_output[i]['title']

    # The output will be "answer1 <extra_id_2> answer2 <extra_id_3> answer3", split it by <extra_id_2> and <extra_id_3>

    for entry in sentences:
        current_output = deepcopy(entry['output'])
        entry['output'] = []

        current_output = current_output.split('<extra_id_1>')[1]
        
        entry['output'].append(current_output.split('<extra_id_2>')[0].strip())
        entry['output'].append(current_output.split('<extra_id_2>')[1].split('<extra_id_3>')[0].strip())
        entry['output'].append(current_output.split('<extra_id_3>')[1].strip())
    
    return sentences[0]['output']


if __name__ == "__main__":
    args = parse_args()
    question = input('Question: ')
    answer = input('Answer: ')
    sentences = main(question, answer, args.model_name_or_path)
    print(sentences)