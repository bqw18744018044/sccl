# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-01 20:42:16
LastEditTime: 2022-04-01 21:36:52
LastEditors: Qiangwei Bai
FilePath: /DECBert/encoder.py
Description: 
"""
import torch

from tqdm import tqdm
from torch import Tensor
from numpy import ndarray
from typing import Union, List
from transformers import AutoModel, AutoTokenizer


class SenEncoder():
    def __init__(self, model = None, tokenzier = None, device: str = None, pooler: str = "avg"):
        self.model = model
        self.tokenizer = tokenzier

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if pooler is not None:
            self.pooler = pooler
        else:
            self.pooler = "avg"
    
    def encode(self,
               sentence: Union[str, List[str]],
               device: str = None,
               return_numpy: bool = False,
               normalize_to_unit: bool = False,
               keepdim: bool = False,
               batch_size: int = 32,
               max_length: int = 128) -> Union[ndarray, Tensor]:
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size: (batch_id + 1)*batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooler == "avg":
                    attention_mask = inputs["attention_mask"]
                    attention_mask = attention_mask.unsqueeze(-1)
                    embeddings = torch.sum(outputs.last_hidden_state*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                else:
                    raise NotImplementedError()

                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())

        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


if __name__ == "__main__":
    pass