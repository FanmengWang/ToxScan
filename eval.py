# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
from multiprocessing import Pool
from tqdm import tqdm, trange
import sys
import pickle
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score, f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, average_precision_score
from scipy.stats import pearsonr


if __name__ == '__main__':
    
    conf_size = 11
    classification_num = 26
    regression_num = 5
        
    classification_target_pad_idx = -10000
    regression_target_pad_idx = -10000.0
    
    title = ['Carcinogenicity', 'Ames Mutagenicity', 
             'Respiratory toxicity', 
             'Eye irritation', 'Eye corrosion', 
             'Cardiotoxicity1', 'Cardiotoxicity10', 'Cardiotoxicity30', 'Cardiotoxicity5', 
             'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4', 
             'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
             'Acute oral toxicity (LD50)', 'LC50DM', 'BCF', 'LC50', 'IGC50']
    
    classification_title = ['Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity', 'Eye irritation', 'Eye corrosion', 'Cardiotoxicity1', 'Cardiotoxicity10', 'Cardiotoxicity30', 'Cardiotoxicity5', 'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    regression_title = ['Acute oral toxicity (LD50)', 'LC50DM', 'BCF', 'LC50', 'IGC50']
        
    predicts_paths = []
    for path_id in range(10):
        path = f'./infer_drug_toxicity_results/infer_drug_toxicity_results_epoch_{path_id}/drug_toxicity_ckpt_epoch_{path_id}_test_cpu.out.pkl'
        predicts_paths.append(path)
    
    result_epoch = 0
    result_pd = pd.DataFrame(columns=title)
    
    for predicts_path in predicts_paths:
            
        predicts = pd.read_pickle(predicts_path)
        
        print(f'-----------------------------epoch:{result_epoch}-----------------------------')
        print(predicts_path)
        
        classification_target = {}
        classification_pred = {}
        for id in range(len(classification_title)):
            target_list = []
            pred_list = []
            title = classification_title[id]
            for epoch in range(len(predicts)):
                predict = predicts[epoch]
                bsz = predicts[epoch]['bsz']
                target_list.append(predict['classfication_target'][:, id])
                pred_list.append(predict['classification_logit_output'][:, id])
                
            target = torch.cat(target_list, dim=0)
            pred = torch.cat(pred_list, dim=0)
            
            masked_tokens = target.ne(classification_target_pad_idx)
            target = target[masked_tokens]
            pred = pred[masked_tokens]

            classification_target[title] = target.float()
            classification_pred[title] = torch.sigmoid(pred.float())


        regression_target = {}
        regression_pred = {}
        for id in range(len(regression_title)):
            target_list = []
            pred_list = []
            title = regression_title[id]
            for epoch in range(len(predicts)):
                predict = predicts[epoch]
                bsz = predicts[epoch]['bsz']
                target_list.append(predict['regression_target'][:, id])
                pred_list.append(predict['regression_logit_output'][:, id])
                
            target = torch.cat(target_list, dim=0)
            pred = torch.cat(pred_list, dim=0)
            
            masked_tokens = target.ne(regression_target_pad_idx)
            target = target[masked_tokens]
            pred = pred[masked_tokens]

            regression_target[title] = target.float()
            regression_pred[title] = pred.float()
            
                    
        sota_num = 0
        scores = []
        
        # classification_task
        for id in range(len(classification_target)):
            title = classification_title[id]
            y_true = classification_target[title].view(-1, conf_size).numpy().mean(axis=1)
            y_pred = classification_pred[title].view(-1, conf_size).numpy().mean(axis=1)
            
            # ACC
            # y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
            # scores.append(round(accuracy_score(y_true, y_pred), 4))
                 
            # AUC       
            roc_auc = roc_auc_score(y_true, y_pred)
            scores.append(roc_auc)
        
        
        # regression_task
        for id in range(len(regression_target)):
            title = regression_title[id]
            y_true = regression_target[title].view(-1, conf_size).numpy().mean(axis=1)
            y_pred = regression_pred[title].view(-1, conf_size).numpy().mean(axis=1)
            
            # RMSE
            # scores.append(np.sqrt(np.mean(np.square(y_pred - y_true))))
            
            # R2
            r2 = r2_score(y_true, y_pred)
            scores.append(r2)
        
        result_pd.loc[result_epoch] = scores
        result_epoch = result_epoch + 1
    
    
    eval_result_path = './eval_result'
    if not os.path.exists(eval_result_path):
        os.makedirs(eval_result_path)
    
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    result_pd.to_csv(f"./{eval_result_path}/eval_result_AUCR2_{time_str}.csv", index=None)
    
    
    
    
    

    
    