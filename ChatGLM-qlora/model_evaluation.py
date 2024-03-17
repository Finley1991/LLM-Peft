#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/16 下午9:34
# @Author  : Finley
# @File    : model_evaluation.py
# @Software: PyCharm
import json
import torch
from transformers import AutoTokenizer
from component.utils import ModelUtils

MODEL_NAME_OR_PATH = '/mnt/wangyafan/project/LLM-quickstart/hf/ZhipuAI/chatglm3-6b'
ADAPTER_NAME_OR_PATH = ('/mnt/wangyafan/project/LLM-Peft/ChatGLM-qlora/trained_models/'
                        'chatglm-zhouyi/final')
TEST_FILE = '/mnt/wangyafan/project/LLM-Peft/ChatGLM-qlora/data/zhouyi_dataset_test.csv'
TEST_RESULT_FILE = '/mnt/wangyafan/project/LLM-Peft/ChatGLM-qlora/evaluation/zhouyi_dataset_test_result.csv'


def main():
    # 使用base model和adapter进行推理，无需手动合并权重
    model_name_or_path = MODEL_NAME_OR_PATH
    adapter_name_or_path = ADAPTER_NAME_OR_PATH

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False

    # 生成超参配置
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda'

    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )

    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    # if tokenizer.__class__.__name__ == 'QWenTokenizer':
    #     tokenizer.pad_token_id = tokenizer.eod_id
    #     tokenizer.bos_token_id = tokenizer.eod_id
    #     tokenizer.eos_token_id = tokenizer.eod_id
    bos_token_id = 64792  # sop
    mask_token_id = 64790  # gMask

    tokenizer.padding_side = "left"

    with open(TEST_FILE, 'r', encoding='utf-8') as read_file, \
            open(TEST_RESULT_FILE, 'w+', encoding='utf-8') as write_file:
        for line in read_file.readlines()[1:]:
            line = line.strip()
            line_split = line.split(",")
            if len(line_split) != 2:
                continue
            text = line_split[0]
            result = line_split[1]
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            mask_token_id = torch.tensor([[mask_token_id]], dtype=torch.long).to(device)
            bos_token_id = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
            eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
            input_ids = torch.concat([mask_token_id, bos_token_id, input_ids, eos_token_id], dim=1)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

            outputs = outputs.tolist()[0][len(input_ids[0]):]
            response = tokenizer.decode(outputs)
            response = response.strip().replace(tokenizer.eos_token, "").strip()
            write_file.write(text.replace('\n', '。') + '\t' + "ORI:" + result + '\t' + "GLM:" + response + '\n')


if __name__ == '__main__':
    main()
