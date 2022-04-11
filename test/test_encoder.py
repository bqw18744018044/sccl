# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-01 21:12:15
LastEditTime: 2022-04-01 21:38:49
LastEditors: Qiangwei Bai
FilePath: /DECBert/test/test_encoder.py
Description: 
"""
import sys
sys.path.append(".")

import unittest

from torch import Tensor
from numpy import ndarray
from encoder import SenEncoder
from transformers import AutoModel, AutoTokenizer

class TestEncoder(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def testEncode(self):
        model_path = "./data/bert-base-uncased"
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encoder = SenEncoder(model, tokenizer)
        embed1 = encoder.encode(["Hello", "World"])
        embed2 = encoder.encode(["Hello", "World"], return_numpy=True)
        self.assertIsInstance(embed1, Tensor)
        self.assertIsInstance(embed2, ndarray)
        self.assertEqual(embed1.shape[0], 2)

if __name__ == "__main__":
    unittest.main()

