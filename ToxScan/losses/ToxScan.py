# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.data import Dictionary
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import torch.distributed as dist


@register_loss("ToxScan")
class ToxScanLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        
        self.classification_num = 26
        self.regression_num = 5
        
        self.target_pad_idx = -10000.0
        self.classification_target_pad_idx = -10000
        self.regression_target_pad_idx = -10000.0
    
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
                
        bsz = sample['target']['classification_target'].size(0)
        sample_size = bsz
        
        classfication_masked_tokens = sample['target']['classification_target'].ne(self.classification_target_pad_idx)
        classfication_sample_size = classfication_masked_tokens.long().sum() 
         
        regression_masked_tokens = sample['target']['regression_target'].ne(self.regression_target_pad_idx)
        regression_sample_size = regression_masked_tokens.long().sum() 

        loss = torch.zeros(1).to(sample['target']['classification_target'].device)
        classification_loss = torch.zeros(1).to(sample['target']['classification_target'].device)
        regression_loss = torch.zeros(1).to(sample['target']['classification_target'].device)
        Ed_classification_loss = torch.zeros(1).to(sample['target']['classification_target'].device)
        
        if classfication_sample_size != 0:
            classfication_target = sample['target']['classification_target'][classfication_masked_tokens]
            classification_logit_output = net_output[0][classfication_masked_tokens]
            classification_loss = self.compute_loss(classification_logit_output, classfication_target, reduce=reduce)
            loss = loss + classification_loss
        
        
        Ed_classfication_masked_tokens = sample['target']['classification_target'][:, -10:].ne(self.classification_target_pad_idx)
        Ed_classfication_sample_size = Ed_classfication_masked_tokens.long().sum() 
        if Ed_classfication_sample_size != 0:
            Ed_classfication_target = sample['target']['classification_target'][:, -10:][Ed_classfication_masked_tokens]
            Ed_classification_logit_output = net_output[0][:, -10:][Ed_classfication_masked_tokens]
            Ed_classification_loss = self.compute_loss(Ed_classification_logit_output, Ed_classfication_target, reduce=reduce)
            loss = loss + Ed_classification_loss * 2
        
        if regression_sample_size !=0:
            regression_target = sample['target']['regression_target'][regression_masked_tokens]
            regression_logit_output = net_output[1][regression_masked_tokens]
            regression_loss = F.l1_loss(
                        regression_logit_output.view(-1).float(),
                        regression_target.view(-1).float(),
                        reduction="mean",
            )
            loss = loss + regression_loss * 2
        
        if not self.training:
            logging_output = {
                "loss": loss.data,
                "classification_loss": classification_loss.data,
                "regression_loss": regression_loss.data,
                
                "classfication_sample_size": classfication_sample_size,
                "regression_sample_size": regression_sample_size,
                "sample_size": sample_size,
                "bsz": bsz,
                
                "classification_logit_output": net_output[0].data,
                "regression_logit_output": net_output[1].data,
                "classfication_target": sample['target']['classification_target'].data,
                "regression_target": sample['target']['regression_target'].data,
                
                "smi_name": sample["smi_name"]
            }
                
        else:
            logging_output = {
                "loss": loss.data,
                "classification_loss": classification_loss.data,
                "regression_loss": regression_loss.data,
                
                "classfication_sample_size": classfication_sample_size,
                "regression_sample_size": regression_sample_size,
                "sample_size": sample_size,
                "bsz": bsz,
            }
            
        return loss, sample_size, logging_output
    
     
    def compute_loss(self, logits, targets, reduce=True):
        logits = logits.float()
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean" if reduce else "none",
        )
        return loss


    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        classification_loss_sum = sum(log.get("classification_loss", 0) for log in logging_outputs)
        Ed_classification_loss_sum = sum(log.get("Ed_classification_loss", 0) for log in logging_outputs)
        Ed_classification_loss2_sum = sum(log.get("Ed_classification_loss2", 0) for log in logging_outputs)
        regression_loss_sum = sum(log.get("regression_loss", 0) for log in logging_outputs)
        
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "classification_loss", classification_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "regression_loss", regression_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        if "valid" in split or "test" in split:
            classification_target_pad_idx = -10000
            regression_target_pad_idx = -10000.0
            
            classfication_target = torch.cat([log.get("classfication_target") for log in logging_outputs], dim=0)
            regression_target = torch.cat([log.get("regression_target") for log in logging_outputs], dim=0)
            classification_logit_output = torch.cat([log.get("classification_logit_output") for log in logging_outputs], dim=0)
            regression_logit_output = torch.cat([log.get("regression_logit_output") for log in logging_outputs], dim=0)
            
            classfication_masked_tokens = classfication_target.ne(classification_target_pad_idx)
            classfication_target = classfication_target[classfication_masked_tokens]
            classfication_pred = classification_logit_output[classfication_masked_tokens]
            
            regression_masked_tokens = regression_target.ne(regression_target_pad_idx)
            regression_target = regression_target[regression_masked_tokens]
            regression_pred = regression_logit_output[regression_masked_tokens]
             
            roc_auc = roc_auc_score(classfication_target.float().cpu().numpy(), torch.sigmoid(classfication_pred.float()).cpu().numpy())
            r2 = r2_score(regression_target.float().cpu().numpy(), regression_pred.float().cpu().numpy())
                        
            score = roc_auc + r2
            metrics.log_scalar(f"{split}_agg_auc", score, sample_size, round=3)
            


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
        