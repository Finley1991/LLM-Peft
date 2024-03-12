#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 下午9:39
# @Author  : Finley
# @File    : train_qlora.py
# @Software: PyCharm
import os
import argparse
import torch
import bitsandbytes as bnb
from loguru import logger
from os.path import join
from collections import defaultdict
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)
from component.collator import SFTDataCollator
from component.dataset import SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss


def verify_model_dtype(model):
    dtype2param_num = defaultdict(int)
    dtype2param_name = defaultdict(list)
    dtype2trainable_param_num = defaultdict(int)
    dtype2trainable_param_name = defaultdict(list)
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)
    print()

    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    Find all linear layer names in the model
    :param model:
    :return:
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names)==1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_args_file", type=str, default="config/config.json", help="The path to the training arguments file"
    )
    args = parser.parse_args()
    train_args_file = args.train_args_file
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    set_seed(training_args.seed)
    return args, training_args


def init_components(args, training_arges):
    """
    初始化自定义组件
    :param args:
    :param training_arges:
    :return:
    """
    # 多卡
    # world_size = 2
    # ddp = True
    # device_map = "auto"
    # if os.environ.get("LOCAL_RANK") is not None:
    #     local_rank = int(os.environ.get("LOCAL_RANK", '0'))
    #     device_map = {"": local_rank}
    training_arges.ddp_find_unused_parameters = False
    local_rank = int(os.environ.get("LOCAL_RANK", '0'))
    device_map = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == "llama" else True

    )

    # Qwen Tokenizer比较特殊，pad_token_id bos_token_id eos_token_id 均为None，eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == "QwenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_arges.gradient_checkpointing
    )

    print(f"memory usage: {model.get_memory_footprint()/(1024*1024*1024)} GB")
    target_modules = find_all_linear_names(model)

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"

    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    verify_model_dtype(model)

    loss_fn = TargetLMLoss(ignore_index=-100)

    train_dataset = SFTDataset(
        args.train_file,
        tokenizer,
        args.max_seq_length
    )
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    trainer = LoRATrainer(
        model=model,
        args=training_arges,
        data_collator=data_collator,
        compute_loss=loss_fn
    )

    return trainer


def main():
    args, training_args = setup_everything()
    trainer = init_components(args, training_args)
    logger.info("Start training")
    train_result = trainer.train()
    logger.info("Save the best checkpoint")
    trainer.save_model(join(training_args.output_dir, "best_checkpoint"))
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

