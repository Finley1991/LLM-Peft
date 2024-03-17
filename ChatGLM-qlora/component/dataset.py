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
        # GLM中的特殊token_id，需要放在input之前,
        self.glm_bos_token_id = 64792
        self.glm_gmask_token_id = 64790
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        # 去掉首行以及前后空格
        self.data_list = [x for x in data_list[1:] if len(x.split(",")) == 2]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 数据格式：question,answer
        data = self.data_list[index]
        question, answer = data.split(",")
        utterances = [question, answer]
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids
        # glm模型输入需要加上特殊token，[gmask, sop],[64790, 64792]
        input_ids = [self.glm_gmask_token_id, self.glm_bos_token_id]
        labels = [0, 0]
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            # 用于对input进行mask，只计算target部分的loss
            if i % 2 == 0:
                labels += [0] * (len(utterances_id) + 1)
            else:
                labels += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(labels)
        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(labels) == len(attention_mask)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return inputs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/wangyafan/project/LLM-quickstart/hf/ZhipuAI/chatglm3-6b",
        trust_remote_code=True,
        use_fast=True
    )
    print(tokenizer.bos_token_id, tokenizer.bos_token,
          tokenizer.eos_token_id, tokenizer.eos_token,
          tokenizer.pad_token_id, tokenizer.pad_token)
    print(tokenizer.encode("</s>"))
    print(tokenizer.decode([64790, 64792, 2893, 30917, 30994]))
    dataset = SFTDataset(
        "/mnt/wangyafan/project/LLM-Peft/ChatGLM-qlora/data/zhouyi_dataset_train.csv",
        tokenizer,
        1024
    )
    print(dataset[0])
