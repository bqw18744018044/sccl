# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-03-26 15:18:29
LastEditTime: 2022-04-03 09:35:07
LastEditors: Qiangwei Bai
FilePath: /DECBert/loss.py
Description: 
"""
import torch
import torch.nn as nn

class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(size_average=False)

    def _target_distribution(self, cluster_prob):
        weight = (cluster_prob ** 2) / (torch.sum(cluster_prob, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, cluster_prob):
        target = self._target_distribution(cluster_prob).detach()
        loss = self.kl_loss((cluster_prob + 1e-08).log(), target) / cluster_prob.shape[0]
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temp, device):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.device = device
        self.cos = nn.CosineSimilarity(dim=-1)
        self.ce_loss_func = nn.CrossEntropyLoss()

    def forward(self, feature1, feature2):
        assert feature1.shape == feature2.shape
        cos_sim = self.cos(feature1.unsqueeze(1), feature2.unsqueeze(0))/self.temp
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss = self.ce_loss_func(cos_sim, labels)
        return loss