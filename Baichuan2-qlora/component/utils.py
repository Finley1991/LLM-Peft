#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 下午9:21
# @Author  : Finley
# @File    : utils.py
# @Software: PyCharm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel


class ModelUtils(object):
    @classmethod
    def load_model(cls, model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            quantization_config = None

        # 加载base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=quantization_config
        )

        # 加载adapter
        if adapter_name_or_path is not None:
            model = PeftModel.from_pretrained(model, adapter_name_or_path)

        return model