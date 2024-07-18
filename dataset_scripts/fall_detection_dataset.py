import os
import pandas as pd
import numpy as np
import sparse
import pickle

import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm


np.random.seed(0)

chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height, width = 260, 346
size = 0.25


# Source data folder
path_dataset = '../datasets/Fall Detection Dataset/'
files = os.listdir(path_dataset + 'aedat4 version/')
parser = aermanager.parsers.parse_aedat4 

# Target data folder
if not os.path.isdir(path_dataset + 'Fall_Detection_splits'):
    os.mkdir(path_dataset + 'Fall_Detection_splits')
    os.makedirs(path_dataset + 'Fall_Detection_splits/' + f'dataset_4sets_{chunk_len_us}/train')
    os.makedirs(path_dataset + 'Fall_Detection_splits/' + f'dataset_4sets_{chunk_len_us}/test')
 
# Init train samples and test samples
train_samples_4sets, test_samples_4sets = [], []

# Pre-defined train samples list and test samples list
test_4sets_list = path_dataset + 'test_4sets_list.txt'
train_4sets_list = path_dataset + 'train_4sets_list.txt'

# If samples are given:
f = open(train_4sets_list, 'r')
for index in f:
    train_samples_4sets.append(index[:-1])
f.close()
f = open(test_4sets_list, 'r')
for index in f:
    test_samples_4sets.append(index[:-1])
f.close()



for events_file in tqdm(files):
    shape, events = load_events_from_file(path_dataset + 'aedat4 version/' + events_file, parser=parser)
    labels = pd.read_csv(path_dataset + 'labels/' + events_file.replace('.aedat4', '.csv'))
    filename_dst = path_dataset + 'Fall_Detection_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
        events_file.replace('.aedat4', '_{}_{}.pckl')

    for _,row in labels.iterrows():
    
        print('events_file = ', events_file)
        startTime_event_num = np.where(events['t']==row.startTime_ev)[0][0]
        endTime_event_num = np.where(events['t']==row.endTime_ev)[0][0]
        sample_events = events[startTime_event_num:endTime_event_num]
        
        total_events = np.array([sample_events['x'], sample_events['y'], sample_events['t'], sample_events['p']]).transpose()

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
        breakpoint()
        total_frames = sparse.stack(total_frames)
        
        total_frames = np.clip(total_frames, a_min=0, a_max=255)
        total_frames = total_frames.astype('uint8')

        if '_4' or '-4' in events_file:  val_set = 'S4' 
        elif '_3' or '-3' in events_file:  val_set = 'S3' 
        elif '_2' or '-2' in events_file:      val_set = 'S2' 
        elif '_1' or '-1' in events_file:    val_set = 'S1'
        else: raise ValueError('Set not handled')
        
        
        
        if events_file in train_samples_4sets:
            pickle.dump(total_frames, open(filename_dst.format('4sets', 'train', val_set, row['class']), 'wb'))
        if events_file in test_samples_4sets:
            pickle.dump(total_frames, open(filename_dst.format('4sets', 'test', val_set, row['class']), 'wb'))
