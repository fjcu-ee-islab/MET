import os
import pandas as pd
import numpy as np
import sparse
import pickle
import scipy.io
import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import random
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

# Used for .mat from spiking_motion_layer1 event output
def spiking_motion_layer1_mat_to_events(filename):
    f = scipy.io.loadmat(filename, mat_dtype=True)

    layer1_x = f['layer1_output']['x']
    layer1_y = f['layer1_output']['y']
    layer1_ts = f['layer1_output']['ts']
    layer1_sp = f['layer1_output']['sp']
    layer1_dir = f['layer1_output']['dir']
    layer1_p = f['layer1_output']['p']

    x = layer1_x[0][0].astype('uint32').reshape(layer1_x[0][0].size,)
    y = layer1_y[0][0].astype('uint32').reshape(layer1_y[0][0].size,)
    t = layer1_ts[0][0].astype('uint32').reshape(layer1_ts[0][0].size,)
    sp = layer1_sp[0][0].astype('uint32').reshape(layer1_sp[0][0].size,)
    dir = layer1_dir[0][0].astype('uint32').reshape(layer1_dir[0][0].size,)
    p = layer1_p[0][0].astype('uint32').reshape(layer1_p[0][0].size,)

    events_struct = [("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("p", np.uint32), ("sp", np.uint32), ("dir", np.uint32) ]
    motion_event = np.fromiter(zip(x, y, t, p, sp, dir), dtype=events_struct)
    return motion_event

# Used for .mat from spiking_motion_layer2 event output
def spiking_motion_layer2_mat_to_events(filename):
    f = scipy.io.loadmat(filename, mat_dtype=True)

    layer2_x = f['layer2_output']['x']
    layer2_y = f['layer2_output']['y']
    layer2_ts = f['layer2_output']['ts']
    layer2_sp = f['layer2_output']['sp']
    layer2_dir = f['layer2_output']['dir']
    layer2_p = f['layer2_output']['p']

    x = layer2_x[0][0].astype('uint32').reshape(layer2_x[0][0].size,)
    y = layer2_y[0][0].astype('uint32').reshape(layer2_y[0][0].size,)
    t = layer2_ts[0][0].astype('uint32').reshape(layer2_ts[0][0].size,)
    sp = layer2_sp[0][0].astype('uint32').reshape(layer2_sp[0][0].size,)
    dir = layer2_dir[0][0].astype('uint32').reshape(layer2_dir[0][0].size,)
    p = layer2_p[0][0].astype('uint32').reshape(layer2_p[0][0].size,)

    events_struct = [("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("p", np.uint32), ("sp", np.uint32), ("dir", np.uint32) ]
    motion_event = np.fromiter(zip(x, y, t, p, sp, dir), dtype=events_struct)
    return motion_event
    
def nda(data, aug):
    
    rotate = transforms.RandomRotation(degrees=30)
    shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))
    
    data_dense = data.todense()
    data_tensor = torch.from_numpy(data_dense)
    data_tensor = data_tensor.permute([0, 3, 1, 2])
    
    if aug == 'roll':
        off1 = random.randint(-5, 5)
        off2 = random.randint(-5, 5)
        data1 = torch.roll(data_tensor, shifts=(off1, off2), dims=(2, 3))
    if aug == 'rotate':
        data1 = rotate(data_tensor)
    if aug == 'shear':
        data1 = shearx(data_tensor)

    data2 = data1.permute([0, 2, 3, 1])
    data2_npy = data2.numpy()
    data2_c00 = sparse.COO(data2_npy)
    
    return data2_c00
