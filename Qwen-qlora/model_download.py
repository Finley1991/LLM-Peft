#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 下午9:34
# @Author  : Finley
# @File    : model_download.py
# @Software: PyCharm
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('qwen/Qwen-14B', cache_dir='/mnt/wangyafan/project/LLM-quickstart/hf/', revision='v1.0.8')