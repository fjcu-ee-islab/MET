import torch
from data_generation import Event_DataModule

from pytorch_lightning.metrics import Accuracy
import pandas as pd
import json
from tqdm import tqdm
import time
import numpy as np

import evaluation_heatmap_utils
from trainer import EvNetModel

import matplotlib.pyplot as plt
from scipy.io import savemat

device = 'cuda:0'
# device = 'cpu'


# DVS128_10class: SME
# path_model = './visual/SME_train/DVSGesture_SME_10classes_9969'
# DVS128_10class: OF
# path_model = './visual/SME_train/DVSGesture_10c_OF_320epoch_9858_lr0001'


# DVS128_11class: SME
# path_model = './visual/SME_train/DVSGesture_SME_11classes_9749'
# DVS128_11class: OF
# path_model = './nvme1n1/MotionEventTransformer/pretrained_models/tests/1110_1056_model_678_DVSGesture_11c_OF_320epoch_9664'


# SL_Animals_DVS_4s: Original
# path_model = './visual/SME_train/SL_Animals_DVS_4s_Ori_8912'
# SL_Animals_DVS_4s: SME
# path_model = './visual/SME_train/20240410/0323_0959_model_8_SLAnimals_4s_SME_72ms_0.9695'
path_model = './visual/SME_train/20240206/SL_Animals_DVS_4s_SME_9808'
# SL_Animals_DVS_4s: OF
# path_model = './visual/SME_train/SL_Animals_DVS_4s_ERAFT_9179'


# SL_Animals_DVS_3s: Original
# path_model = './visual/SME_train/SL_Animals_DVS_3s_Ori_640epoch_8852'

# IITM: Original
# path_model = './visual/SME_train/IITM_EVT_9909'
# IITM: SME
# path_model = './visual/SME_train/IITM_SME_9999'
# IITM: OF
# path_model = './visual/SME_train/IITM_OF_9999'

# UCF11: Original
# path_model = 
# UCF11: SME
# path_model = './visual/SME_train/UCF11_SME_8654'
# UCF11: OF
# path_model = './visual/SME_train/0206_0344_model_688_UCF11_OF'
# path_model = './nvme1n1/MotionEventTransformer/pretrained_models/tests/0216_2330_model_688'

# FDD: Original
# path_model = './visual/SME_train/FDD_EVT_9333'
# FDD: SME
# path_model = './visual/SME_train/FDD_SME_9999'
# FDD: OF
# path_model = './visual/SME_train/0204_1541_model_702_FDD_OF'
# path_model = './visual/SME_train/0204_1943_model_703_FDD_OF'

path_weights = evaluation_heatmap_utils.get_best_weigths(path_model, 'val_acc', 'max')
#evaluation_heatmap_utils.plot_training_evolution(path_model)
all_params = json.load(open(path_model + '/all_params.json', 'r'))
model = EvNetModel.load_from_checkpoint(path_weights, map_location=torch.device('cpu'), **all_params).eval().to(device)

evaluation_heatmap_utils.get_evaluation_results(path_model, path_weights)

def get_params(model):
    total_params = pd.DataFrame([ (n.split('.')[0],p.numel()/1000000) for n,p in model.backbone.named_parameters() if p.requires_grad ]).groupby(0).sum().sum().iloc[0]
    pos_encoding_params = pd.DataFrame([ (n.split('.')[0],p.numel()/1000000) for n,p in model.backbone.named_parameters() if p.requires_grad ]).groupby(0).sum().loc['pos_encoding'].iloc[0]
    stats = {
        'total_params': total_params,
        'backbone_params': total_params - pos_encoding_params,
        'pos_encoding_params': pos_encoding_params
        }
    return stats



def get_complexity_stats(model, all_params):
    data_params = all_params['data_params']
    data_params['batch_size'] = 1
    data_params['pin_memory'] = False
    data_params['sample_repetitions'] = 1
    dm = Event_DataModule(**data_params)
    dl = dm.val_dataloader()
    
    # https://github.com/sovrasov/flops-counter.pytorch
    from ptflops import get_model_complexity_info
    
    total_flops, total_macs, total_params, total_act_patches = [], [], [], []
    total_time_flops = []
    counter = 0
    for polarity, pixels, labels in tqdm(dl):
        if polarity is None: continue
        polarity, pixels, labels = polarity.to(device), pixels.to(device), labels.to(device)
        
        for ts in range(len(polarity)):
            num_patches = sum(polarity[ts:ts+1].sum(-1).sum(0).sum(0) != 0)
            mask = polarity[ts:ts+1].sum(-1).sum(0).sum(0) != 0
            pol_t, pix_t = polarity[ts:ts+1][:,:,mask,:], pixels[ts:ts+1][:,:,mask,:]            
            t = time.time()

            macs, params = get_model_complexity_info(model.backbone, 
                                               ({'kv': pol_t, 'pixels': pix_t},),
                                             input_constructor=lambda x: x[0],
                                             as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
            total_time_flops.append(time.time() - t)
            flops = 2*macs
            total_act_patches.append(num_patches.cpu())
        
        total_flops.append(flops); total_macs.append(macs); total_params.append(params)
        
        
    return np.mean(total_flops), np.mean(total_act_patches), np.mean(total_macs)



# %%

# =============================================================================
# Time analysis
# =============================================================================

def get_time_accuracy_stats(model, all_params):
    data_params = all_params['data_params']
    data_params['batch_size'] = 1
    data_params['pin_memory'] = False
    data_params['sample_repetitions'] = 1
    dm = Event_DataModule(**data_params)
    dl = dm.val_dataloader()
        
    total_time = []
    y_true, y_pred = [], []
    for polarity, pixels, labels in tqdm(dl):
        if polarity is None: continue
        polarity, pixels, labels = polarity.to(device), pixels.to(device), labels.to(device)
        t = time.time()
        embs, clf_logits = model(polarity, pixels)
        total_time.append((time.time() - t)/len(polarity))
        
        y_true.append(labels[0])
        y_pred.append(clf_logits.argmax())

    y_true, y_pred = torch.stack(y_true).to("cpu"), torch.stack(y_pred).to("cpu")
    acc_score = Accuracy()(y_true, y_pred).item()
    
    logs = evaluation_heatmap_utils.load_csv_logs_as_df(path_model)
    train_acc = logs[~logs['val_acc'].isna()]['val_acc'].max()

    return np.mean(total_time)*1000, train_acc, acc_score



