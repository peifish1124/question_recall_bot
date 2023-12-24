from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是能根據文章來生出問題、選項以及對應答案的專家，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、明確、詳細的回答。\
            USER: 請針對以下文章生出一個問題、四個選項以及一個正確答案，且需要確保四個選項中只有一個是正確答案，而其他三個是錯誤答案。 文章:{instruction} ASSISTANT:"
def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
 #           load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
    )
    return quantization_config
