# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-03-24 21:23:00
LastEditTime: 2022-03-24 21:23:01
LastEditors: Qiangwei Bai
FilePath: /deepcn/logger.py
Description: 
"""
import sys
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
