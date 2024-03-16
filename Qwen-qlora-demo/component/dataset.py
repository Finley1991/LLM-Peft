#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 下午9:23
# @Author  : Finley
# @File    : dataset.py
# @Software: PyCharm
import json
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 数据格式：<s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data["conversation"]

        # 收集多轮对话
        utterances = []
        for x in conversation:
            utterances.append(x["human"])
            utterances.append(x["assistant"])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids
        input_ids = [self.bos_token_id]
        target_mask = [0]
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            # 用于对input进行mask，只计算target部分的loss
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask
        }
        return inputs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/wangyafan/project/LLM-quickstart/hf/qwen/Qwen-14B",
        trust_remote_code=True,
        use_fast=True
    )
    dataset = SFTDataset(
        "/mnt/wangyafan/project/LLM-Peft/Qwen-qlora-demo/data/text_matching_data_train.jsonl",
        tokenizer,
        1024
    )
    print(dataset[0])