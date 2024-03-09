#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 下午8:25
# @Author  : Finley
# @File    : argument.py
# @Software: PyCharm
from dataclasses import dataclass,field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    自定义参数
    """
    max_seq_length: int = field(metadata={"help:输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "评测数据集"})


@dataclass
class QLoRAArguments:
    """
    自定义参数
    """
    max_seq_length: int = field(metadata={"help:输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    task_type: str = field(default="", metadata={"help": "预训练任务：[sft, pretrain]"})
    eval_file: Optional[str] = field(default="", metadata={"help": "评测数据集"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
