#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 上午9:44
# @Author  : Finley
# @File    : loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
class Loss(object):
    """
    Loss class parent class
    """
    def __call__(self, model, inputs, training_args, return_outputs=False):
        """
        Used to calculate loss.
        Looking at the source code, return_outputs=True is called during training, and return_outputs=False is called during eval and predict
        :param model: model
        :param inputs: model input, dict
        :param training_args: training configuration parameters
        :param return_outputs: whether to return the output of the model
        :return:
        """
        raise NotImplemented


class TargetLMLoss(Loss):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, training_args, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        # 模型前向预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss