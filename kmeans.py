# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-01 22:20:05
LastEditTime: 2022-04-05 11:50:56
LastEditors: Qiangwei Bai
FilePath: /DECBert/kmeans.py
Description: 
"""
import torch

from typing import List
from logger import logger
from numpy import ndarray
from metric import Confusion
from encoder import SenEncoder
from dataset import get_dataset
from sklearn.cluster import KMeans


def get_kmeans_center(texts: List[str], num_classes: int, encoder: SenEncoder, labels: List[int] = None) -> ndarray:
    logger.info("-- Initialing the cluster centers with k-means")
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes, random_state=0)
    all_embeddings = encoder.encode(texts)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_
    pred_labels = torch.tensor(cluster_assignment)

    if labels:
        true_labels = torch.tensor(labels)
        logger.info("all_embeddings={}, true_labels={}, pred_labels={}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))

        confusion.add(pred_labels, true_labels)
        confusion.optimal_assignment(num_classes)
        acc = confusion.acc()
        nmi = confusion.clusterscores()["NMI"]
        logger.info("Iterations={}, Clustering ACC={:.3f}, Clustering NMI={}".format(clustering_model.n_iter_, acc, nmi))
    return clustering_model.cluster_centers_, {"ACC": acc, "NMI": nmi}


if __name__ == "__main__":
    pass
