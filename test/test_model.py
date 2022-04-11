# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-01 22:38:29
LastEditTime: 2022-04-03 09:19:52
LastEditors: Qiangwei Bai
FilePath: /DECBert/test/test_model.py
Description: 
"""
import sys
sys.path.append(".")

import unittest
import numpy as np
from model import DECBert
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from dataset import SimpleDataset, get_dataset, DataCollator

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def testModel(self):
        batch_size = 4
        num_classes = 8
        model_path = "./data/bert-base-uncased"
        bert = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        texts, labels = get_dataset("searchsnippets", "test")
        dataset = SimpleDataset(texts=texts, labels=labels, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=DataCollator())
        batch = next(iter(dataloader))
        cluster_centers = np.random.randn(num_classes,768)
        dec = DECBert(backbone=bert, cluster_centers=cluster_centers)
        output = dec(**batch)
        self.assertGreater(output.loss.item(), 0)
        self.assertEqual(output.cluster_prob.shape[0], batch_size)
        self.assertEqual(output.cluster_prob.shape[1], num_classes)


if __name__ == "__main__":
    unittest.main()