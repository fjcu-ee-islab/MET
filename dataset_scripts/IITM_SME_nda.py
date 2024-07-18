import os
import pandas as pd
import numpy as np
import sparse
import pickle
import glob

import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import scipy.io
import my_nda_aug

mapping = { 0 :'comeHere',
            1 :'left_swipe',
            2 :'right_swipe',
            3 :'rotation_CCW',
            4 :'rotation_CW',
            5 :'swipeDown',
            6 :'swipeUP',
            7 :'swipeV',
            8 :'X',
            9 :'Z'}

def IITM_SME_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("IITM Dataset not found, looked at: {}".format(dataset_path))

    train_files = []
    test_files = []
    for digit in range(10):
        digit_txt_train = glob.glob(os.path.join(dataset_path, 'Train/{}/*.txt'.format(digit)))
        digit_mat_train = glob.glob(os.path.join(dataset_path, 'Train/{}/*.mat'.format(digit)))
        digit_txt_test = glob.glob(os.path.join(dataset_path, 'Test/{}/*.txt'.format(digit))) 

        train_files.append(digit_txt_train)
        train_files.append(digit_mat_train)
        test_files.append(digit_txt_test)
    
    # We need the same number of train and test samples for each digit, let's compute the minimum
    max_n_train = min(map(lambda l: len(l), train_files))
    max_n_test = min(map(lambda l: len(l), test_files))
    n_train = max_n_train # we could take max_n_train, but my memory on the shared drive is full
    n_test = max_n_test # we test on the whole test set - lets only take 100*10 samples
    assert((n_train <= max_n_train) and (n_test <= max_n_test)), 'Requested more samples than present in dataset'

    print("IITM: {} train samples and {} test samples per digit (max: {} train and {} test)".format(n_train, n_test, max_n_train, max_n_test))
    # Crop extra samples of each digits
    train_files = map(lambda l: l[:n_train], train_files)
    test_files = map(lambda l: l[:n_test], test_files)

    return list(train_files), list(test_files)

def IITM_load_events_from_txt(file_path, max_duration=None):
    data = np.loadtxt(file_path)
    return data

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

np.random.seed(0)

chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height, width = 128, 128
size = 0.25


# Source data folder
path_dataset = '../datasets/IITM_VM/'
fns_train, fns_test = IITM_SME_get_file_names(path_dataset+'/all_txt_SME')
fns_train = [val for sublist in fns_train for val in sublist]
fns_test = [val for sublist in fns_test for val in sublist]
files = fns_train+fns_test
train_label_list = [[] for i in range(10)]
test_label_list = [[] for i in range(10)]

for events_file in tqdm(files):
    print('events_file = ', events_file)
    istrain = events_file in fns_train
    root, extension = os.path.splitext(events_file)
    if extension == '.mat':
        if events_file[-11:]=='_layer1.mat':
            label = int(events_file.split('/')[-2])
            motion_events = spiking_motion_layer1_mat_to_events(events_file)
            filename_dst = path_dataset + 'IITM_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
                events_file.split('/')[-1].replace('.mat', '_{}.pckl')

        elif events_file[-11:]=='_layer2.mat':     
            label = int(events_file.split('/')[-2])
            motion_events = spiking_motion_layer2_mat_to_events(events_file)
            filename_dst = path_dataset + 'IITM_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
                events_file.split('/')[-1].replace('.mat', '_{}.pckl')                 
    elif extension == '.txt':
        data = IITM_load_events_from_txt(events_file)
        times = data[:,0] # [ts]
        addrs = data[:,1:] # [p, x, y] 
        label = int(events_file.split('/')[-2])
        filename_dst = path_dataset + 'IITM_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
            events_file.split('/')[-1].replace('.txt', '_{}.pckl')
    
    if extension == '.txt':
        total_events = np.array([addrs[:,1], addrs[:,2], times, addrs[:,0]]).T
    elif extension == '.mat':
        total_events = np.array([motion_events['x'], motion_events['y'], motion_events['t'], motion_events['p'], motion_events['sp'], motion_events['dir']]).transpose()    
    
    total_chunks = []
    
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
        
        # If the numbers of the last events of chunk_len_us (default: 12000 us) <= 4 
        # Only event data (.aedat)
        if total_events.shape[1] == 4:
            if len(chunk_inds) <= 4: 
                pass
            else:
                total_chunks.append(total_events[chunk_inds])
        # Event data with speed and direction (.mat)
        elif total_events.shape[1] == 6:
            if len(chunk_inds) <= 0: 
                pass
            else:
                total_chunks.append(total_events[chunk_inds])
        total_events = total_events[:max(1, chunk_inds.min())-1]

    if len(total_chunks) == 0: continue
    total_chunks = total_chunks[::-1]
    
    total_frames = []
    total_speed = []
    
    for chunk in total_chunks:
        frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                            np.ones(chunk.shape[0]).astype('int32'), 
                            (width, height, 2))   # .to_dense()
        total_frames.append(frame)
        if(chunk.shape[1] == 6):
            frame_speed = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                                chunk[:,4].astype('int32'), 
                                (height, width, 2))   # .todense()
            total_speed.append(frame_speed)

    total_frames = sparse.stack(total_frames)
    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')

    if total_speed != []:
        total_speed = sparse.stack(total_speed)
        total_speed = np.clip(total_speed, a_min=0, a_max=255)
        total_speed = total_speed.astype('uint8')             
    
    if events_file in fns_train:
        train_file_dst = (path_dataset + 'IITM_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/').format('10sets','train')
        os.makedirs(train_file_dst, exist_ok=True)        
        train_file_name = filename_dst.format('10sets','train', label)
        pickle.dump(total_frames, open(train_file_name, 'wb'))
        
        aug_total_frames = my_nda_aug.my_nda_aug(total_frames, 'roll')
        
        pickle.dump(aug_total_frames, open(train_file_name[:115] + 'aug_roll_' + train_file_name[115:], 'wb'))
        aug_total_frames = my_nda_aug.my_nda_aug(total_frames, 'rotate')
        pickle.dump(aug_total_frames, open(train_file_name[:115] + 'aug_rotate_' + train_file_name[115:], 'wb'))
        aug_total_frames = my_nda_aug.my_nda_aug(total_frames, 'shear')
        pickle.dump(aug_total_frames, open(train_file_name[:115] + 'aug_shear_' + train_file_name[115:], 'wb'))

        if(isinstance(total_speed, sparse.COO)):
            speed_filename = (train_file_name[:115]+"speed_"+train_file_name[115:])
            pickle.dump(total_speed, open(speed_filename, 'wb'))

    if events_file in fns_test:
        test_file_dst = (path_dataset + 'IITM_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/').format('10sets','test')
        os.makedirs(test_file_dst, exist_ok=True)
        test_file_name = filename_dst.format('10sets','test', label)
        pickle.dump(total_frames, open(test_file_name, 'wb'))
