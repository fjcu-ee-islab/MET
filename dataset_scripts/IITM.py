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

mapping = { 0 :'comeHome',
            1 :'left_swipe',
            2 :'right_swipe',
            3 :'rotation_CCW',
            4 :'rotation_CW',
            5 :'swipeDown',
            6 :'swipeUP',
            7 :'swipeV',
            8 :'X',
            9 :'Z'}

def IITM_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("IITM Dataset not found, looked at: {}".format(dataset_path))

    train_files = []
    test_files = []
    for digit in range(10):
        digit_train = glob.glob(os.path.join(dataset_path, 'Train/{}/*.txt'.format(digit)))
        digit_test = glob.glob(os.path.join(dataset_path, 'Test/{}/*.txt'.format(digit)))
        train_files.append(digit_train)
        test_files.append(digit_test)

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

np.random.seed(0)

chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height, width = 128, 128
size = 0.25


# Source data folder
path_dataset = '../datasets/IITM/'
fns_train, fns_test = IITM_get_file_names(path_dataset+'/txt')
fns_train = [val for sublist in fns_train for val in sublist]
fns_test = [val for sublist in fns_test for val in sublist]
files = fns_train+fns_test
train_label_list = [[] for i in range(10)]
test_label_list = [[] for i in range(10)]

for events_file in tqdm(files):
    
    istrain = events_file in fns_train
    data = IITM_load_events_from_txt(events_file)
    times = data[:,0] # [ts]
    addrs = data[:,1:] # [p, x, y] 
    label = int(events_file.split('/')[-2])
    
    filename_dst = path_dataset + 'IITM_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
        events_file.split('/')[-1].replace('.txt', '_{}.pckl')
    
    print('events_file = ', events_file)
    
    total_events = np.array([addrs[:,1], addrs[:,2], times, addrs[:,0]]).T
    total_chunks = []
    
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
        if len(chunk_inds) <= 4: 
            pass
        else:
            total_chunks.append(total_events[chunk_inds])
        total_events = total_events[:max(1, chunk_inds.min())-1]
    if len(total_chunks) == 0: 
        print("len(total_chunks) == 0")
        continue
    total_chunks = total_chunks[::-1]
    
    total_frames = []
    
    for chunk in total_chunks:
        frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                            np.ones(chunk.shape[0]).astype('int32'), 
                            (width, height, 2))   # .to_dense()
        total_frames.append(frame)
    total_frames = sparse.stack(total_frames)
    
    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')

    if events_file in fns_train:
        train_file_dst = (path_dataset + 'IITM_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/').format('9sets','train')
        os.makedirs(train_file_dst, exist_ok=True)
        train_file_name = filename_dst.format('10sets','train', label)
        pickle.dump(total_frames, open(train_file_name, 'wb'))
    if events_file in fns_test:
        test_file_dst = (path_dataset + 'IITM_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/').format('9sets','test')
        os.makedirs(test_file_dst, exist_ok=True)
        test_file_name = filename_dst.format('10sets','test', label)
        pickle.dump(total_frames, open(test_file_name, 'wb'))
