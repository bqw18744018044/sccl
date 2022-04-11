# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-05 16:40:50
LastEditTime: 2022-04-06 22:48:56
LastEditors: Qiangwei Bai
FilePath: /SCCL/model.py
Description: 
"""
import torch
import numpy as np
import torch.nn as nn

from numpy import ndarray
from dataclasses import dataclass
from typing import Union, Optional
from torch.nn import functional as F
from loss import ClusterLoss, ContrastiveLoss
from transformers.modeling_outputs import ModelOutput


@dataclass
class SCCLOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embeddings: Optional[torch.FloatTensor] = None
    cluster_prob: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None

class IterSCCL(nn.Module):
    def __init__(self, backbone, cluster_centers: Union[str, ndarray], alpha: float = 1.0):
        super(IterSCCL, self).__init__()
        self.backbone = backbone
        bb_hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(bb_hidden_size, bb_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(bb_hidden_size, 128))
        if isinstance(cluster_centers, str):
            cluster_centers - np.load(cluster_centers)
        initial_cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = nn.Parameter(initial_cluster_centers)
        self.alpha = alpha
        self.cluster_loss = ClusterLoss()
        self.contrastive_loss = ContrastiveLoss(temp=0.05, device=self.backbone.device)

    def  get_embeddings(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            clustering=False):
        if clustering:
            return self.cluter_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            return self.contrastive_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
    
    def cluter_forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        # Instance-CL loss
        embed = self.get_embeddings(input_ids, attention_mask)
        # Clustering loss
        cluster_prob = self.get_cluster_prob(embed)
        cl_loss = self.cluster_loss(cluster_prob)
        return SCCLOutput(
            loss=cl_loss,
            logits=None,
            embeddings=embed,
            cluster_prob=cluster_prob,
            labels=labels
        )

    def contrastive_forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        # Instance-CL loss
        embed0 = self.get_embeddings(input_ids, attention_mask)
        embed1 = self.get_embeddings(input_ids, attention_mask)
        feat1 = F.normalize(self.head(embed0), dim=1)
        feat2 = F.normalize(self.head(embed1), dim=1)
        ct_loss = self.contrastive_loss(feat1, feat2)
        return SCCLOutput(
            loss=ct_loss,
            logits=None,
            embeddings=embed0,
            cluster_prob=None,
            labels=labels
        )


class SCCL(nn.Module):
    def __init__(self, backbone, cluster_centers: Union[str, ndarray], alpha: float = 1.0):
        super(SCCL, self).__init__()
        self.backbone = backbone
        bb_hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(bb_hidden_size, bb_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(bb_hidden_size, 128))
        if isinstance(cluster_centers, str):
            cluster_centers - np.load(cluster_centers)
        initial_cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = nn.Parameter(initial_cluster_centers)
        self.alpha = alpha
        self.cluster_loss = ClusterLoss()
        self.contrastive_loss = ContrastiveLoss(temp=0.05, device=self.backbone.device)

    def  get_embeddings(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        # Instance-CL loss
        embed0 = self.get_embeddings(input_ids, attention_mask)
        embed1 = self.get_embeddings(input_ids, attention_mask)
        feat1 = F.normalize(self.head(embed0), dim=1)
        feat2 = F.normalize(self.head(embed1), dim=1)
        ct_loss = self.contrastive_loss(feat1, feat2)
        # Clustering loss
        cluster_prob = self.get_cluster_prob(embed0)
        cl_loss = self.cluster_loss(cluster_prob)
        loss = ct_loss + cl_loss
        # loss = ct_loss + 0.5*cl_loss
        return SCCLOutput(
            loss=loss,
            logits=None,
            embeddings=embed0,
            cluster_prob=cluster_prob,
            labels=labels
        )


if __name__ == "__main__":
    pass