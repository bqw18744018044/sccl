# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-05 16:40:50
LastEditTime: 2022-04-09 10:33:24
LastEditors: Qiangwei Bai
FilePath: /SCCL/test/test_dataset.py
Description: 
"""
import sys
sys.path.append(".")

import pickle
import unittest
import pandas as pd

from torch import Tensor
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import get_dataset, DataCollator, SimpleDataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def testDataset(self):
        tokenizer = AutoTokenizer.from_pretrained("./data/bert-base-uncased")
        texts, labels, num_classes = get_dataset("SearchSnippets")
        dataset = SimpleDataset(texts=texts, labels=labels, tokenizer=tokenizer)
        sample = dataset[0]
        self.assertEqual(len(sample), 4)
        self.assertIsInstance(sample, dict)
        self.assertIsInstance(sample["input_ids"], Tensor)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=8, collate_fn=DataCollator())
        batch = next(iter(dataloader))
        self.assertIsInstance(batch["input_ids"], Tensor)
        self.assertEqual(batch["input_ids"].shape[0], 8)


if __name__ == "__main__":
    unittest.main()