# -*- coding:utf-8 -*-
"""
Author: Qiangwei Bai
Date: 2022-04-03 09:36:17
LastEditTime: 2022-04-03 09:36:18
LastEditors: Qiangwei Bai
FilePath: /SCCL/trainer.py
Description: 
"""
import sys
import torch
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from typing import Dict
from logger import logger
from metric import Confusion
from encoder import SenEncoder
from model import SCCL, IterSCCL
from sklearn.cluster import KMeans
from kmeans import get_kmeans_center
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalPrediction
from dataset import get_dataset, DataCollator, SimpleDataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer

warnings.filterwarnings("ignore")

class IterSCCLTrainer():
    def __init__(self, args):
        self.args = args

    def train(self):
        logger.info("-"*40)
        logger.info("-"*10 + " Start Training IterSCCL " + "-"*10)
        logger.info("-"*40 + "\n")
        backbone = AutoModel.from_pretrained(self.args.backbone)
        tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        encoder = SenEncoder(backbone, tokenizer)
        logger.info(f"-- Loading dataset: {self.args.dts_name}")
        texts, labels, num_classes = get_dataset(self.args.dts_name)
        logger.info(f"Text number={len(texts)}, Class number={num_classes}")
        if not self.args.num_classes:
            self.args.num_classes = num_classes
        cluster_centers, _ = get_kmeans_center(texts=texts, labels=labels, encoder=encoder, num_classes=self.args.num_classes)
        dataset = SimpleDataset(texts=texts, labels=labels, tokenizer=tokenizer)
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=self.args.train_batch_size, collate_fn=DataCollator())
        eval_dataloader = DataLoader(dataset, shuffle=False, batch_size=self.args.eval_batch_size, collate_fn=DataCollator())
        logger.info(f"-- Training IterSCCL")
        isccl = IterSCCL(backbone=backbone, cluster_centers=cluster_centers, alpha=self.args.alpha)
        isccl = isccl.to("cuda")
        isccl.train()

        optimizer = torch.optim.Adam([
            {'params':isccl.backbone.parameters()}, 
            {'params':isccl.head.parameters(), 'lr': self.args.learning_rate*100},
            {'params':isccl.cluster_centers, 'lr': self.args.learning_rate*100}
            ], lr=self.args.learning_rate)
            
        for epoch in range(self.args.epochs):
            optimizer.zero_grad()
            if epoch%2==0:
                self.train_contrastive_step(epoch, train_dataloader, isccl, optimizer)
            else:
                self.train_cluster_step(epoch, train_dataloader, isccl, optimizer)
            print(self.eval_step(eval_dataloader, isccl))
    
    def eval_step(self, dataloader, model):
        logger.info("Evaluation")
        model.eval()
        embeddings = []
        cluster_prob = []
        labels = []
        for i, batch in tqdm(enumerate(iter(dataloader))):
            with torch.no_grad():
                batch = {k:v.to("cuda") for k,v in batch.items()}
                out = model(**batch, clustering=True)
                embeddings.append(out.embeddings)
                cluster_prob.append(out.cluster_prob)
                labels.append(out.labels)
        model.train()
        embeddings = torch.cat(embeddings)
        cluster_prob = torch.cat(cluster_prob)
        labels = torch.cat(labels)
        preds = cluster_prob.argmax(1)
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        confusion_model.add(preds, labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()
        nmi_model = confusion_model.clusterscores()["NMI"]
        kmeans = KMeans(n_clusters=self.args.num_classes, random_state=0)
        kmeans.fit(embeddings.cpu().numpy())
        kmeans_preds = torch.tensor(kmeans.labels_.astype(np.int64))
        confusion.add(kmeans_preds, labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()
        nmi = confusion.clusterscores()["NMI"]
        metrics = dict()
        metrics["R_NMI"] = nmi  # [Representation] NMI
        metrics["R_ACC"] = acc  # [Representation] ACC
        metrics["M_NMI"] = nmi_model  # [Model] NMI
        metrics["M_ACC"] = acc_model  # acc_model
        return metrics

    def train_cluster_step(self, epoch: str, dataloader, model, optimizer):
        logger.info(f"Epoch={epoch}, Clustering Learning.")
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch = {k:v.to("cuda") for k,v in batch.items()}
            out = model(**batch, clustering=True)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def train_contrastive_step(self, epoch: str, dataloader, model, optimizer):
        logger.info(f"Epoch={epoch}, Contrastive Learning.")
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch = {k:v.to("cuda") for k,v in batch.items()}
            out = model(**batch, clustering=False)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()



class SCCLTrainer():
    def __init__(self, args):
        self.args = args

    def train(self):
        logger.info("-"*40)
        logger.info("-"*10 + " Start Training SCCL " + "-"*10)
        logger.info("-"*40 + "\n")
        backbone = AutoModel.from_pretrained(self.args.backbone)
        tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        encoder = SenEncoder(backbone, tokenizer)
        logger.info(f"-- Loading dataset: {self.args.dts_name}")
        texts, labels, num_classes = get_dataset(self.args.dts_name)
        logger.info(f"Text number={len(texts)}, Class number={num_classes}")
        if not self.args.num_classes:
            self.args.num_classes = num_classes
        dataset = SimpleDataset(texts=texts, labels=labels, tokenizer=tokenizer)
        cluster_centers, _ = get_kmeans_center(texts=texts, labels=labels, encoder=encoder, num_classes=self.args.num_classes)
        logger.info(f"-- Training SCCL")
        sccl = SCCL(backbone=backbone, cluster_centers=cluster_centers, alpha=self.args.alpha)
        training_args = TrainingArguments(output_dir=self.args.output_dir,
                                        num_train_epochs=self.args.epochs,
                                        do_eval=True,
                                        evaluation_strategy=self.args.evaluation_strategy,
                                        per_device_train_batch_size=self.args.train_batch_size,
                                        per_device_eval_batch_size=self.args.eval_batch_size,
                                        logging_steps=self.args.logging_steps,
                                        save_strategy=self.args.evaluation_strategy,
                                        load_best_model_at_end=True,
                                        log_level="error",
                                        metric_for_best_model="M_ACC")
        # optimizer = torch.optim.Adam([
        #    {'params':sccl.backbone.parameters()}, 
        #    {'params':sccl.cluster_centers, 'lr': self.args.learning_rate*100}], lr=self.args.learning_rate)
        optimizer = torch.optim.Adam([
            {'params':sccl.backbone.parameters()}, 
            {'params':sccl.head.parameters(), 'lr': self.args.learning_rate*100},
            {'params':sccl.cluster_centers, 'lr': self.args.learning_rate*100}
            ], lr=self.args.learning_rate)
        collate_fn = DataCollator()
        trainer = Trainer(model=sccl,
                        args=training_args,
                        train_dataset=dataset,
                        eval_dataset=dataset,
                        tokenizer=tokenizer,
                        data_collator=collate_fn,
                        compute_metrics=self.cluster_metrics, 
                        optimizers=(optimizer, None))
        logger.info(trainer.evaluate(dataset))
        trainer.train()
        logger.info(trainer.evaluate(dataset))

    def cluster_metrics(self, eval_output: EvalPrediction) -> Dict[str, float]:
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        embeddings = eval_output.predictions[0]
        cluster_prob = eval_output.predictions[1]
        labels = torch.tensor(eval_output.predictions[2])
        sccl_preds = torch.tensor(cluster_prob.argmax(1))
        confusion_model.add(sccl_preds, labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()
        kmeans = KMeans(n_clusters=self.args.num_classes, random_state=0)
        kmeans.fit(embeddings)
        kmeans_preds = torch.tensor(kmeans.labels_.astype(np.int64))
        confusion.add(kmeans_preds, labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()
        metrics = dict()
        metrics["R_NMI"] = confusion.clusterscores()["NMI"]  # [Representation] NMI
        metrics["R_ACC"] = acc  # [Representation] ACC
        metrics["M_NMI"] = confusion_model.clusterscores()["NMI"]  # [Model] NMI
        metrics["M_ACC"] = acc_model  # acc_model
        return metrics


def get_args(argv):
    parser = argparse.ArgumentParser(
        description='Training and Testing Models.'
    )
    # sup-simcse-bert-base-uncased
    parser.add_argument("--backbone", type=str, default="./data/sup-simcse-bert-base-uncased")
    parser.add_argument("--dts_name", type=str, default="Biomedical")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=28)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./models")

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    # trainer = IterSCCLTrainer(args)
    trainer = SCCLTrainer(args)
    trainer.train()