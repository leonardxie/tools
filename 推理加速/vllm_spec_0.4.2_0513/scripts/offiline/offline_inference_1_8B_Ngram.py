# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------
# @Time   : 2024/2/22 19:14
# @Author : Wang Zibin
# @Feature: 熟悉代码用
# -----------------------------

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import numpy as np
from vllm import LLM, SamplingParams
import timeit
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Sample prompts.
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
TEMPLATE = (
    "<|im_start|>system\n"
    "{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    "你是一个精通古诗和历史的机器人，\n\n"
    "Question: {instruction} <|im_end|>\n"
    "<|im_start|>assistant"
)


def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})


prompts = [
    ("【静夜思：床前明月光，疑是地上霜。举头望明月，低头思故乡。】\n"
     "问题: 背诵《静夜思》全文")
]

inputs = [generate_prompt(text) for text in prompts]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.5,
                                 max_tokens=2000,
                                 stop_token_ids=[151643, 151645],
                                 logits_processors=None,
                                 ngram=None,
                                 whether_use_speculate=False)

# Create an LLM.
llm = LLM(model="/data01/01model_hub/LLM/Qwen-1_8B-Chat",
          draft_model="ngram",
          speculate_length=5,
          gpu_memory_utilization=0.9,
          trust_remote_code=True,
          dtype=torch.float16,
          disable_log_stats=False,
          warmup_length=20,
          )

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start_time = timeit.default_timer()
outputs = llm.generate(inputs, sampling_params)

for output in outputs:
    prompt = output.prompt
    logging.info(f"投机采样结果:\n\n {output.outputs[0].text}")
    evalTime = timeit.default_timer() - start_time
    logging.info(f"  Evaluation done in total {evalTime} secs\n")
    logging.info(f"  Prompt: {prompt!r}\n")
    logging.info(f" 基于transformers底层实现的投机采样速度：{len(output.outputs[0].text) / evalTime} tokens/s")
    mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    logging.info(f" 平均接受长度：{mean_num_accepted}")
