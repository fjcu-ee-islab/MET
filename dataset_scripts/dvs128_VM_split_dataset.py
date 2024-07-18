import aermanager
from aermanager.aerparser import load_events_from_file
import struct
import numpy as np
import scipy.io
import pandas as pd
from tqdm import tqdm
import pickle
import os
import SME

# Source data folder
path_dataset_src = '../datasets/DVS_Gesture_aedat/'
# Target data folder
path_dataset_dst = '../datasets/DvsGesture_SME_nda/clean_dataset/'

train_files, test_files = 'trials_to_train.txt', 'trials_to_test.txt'
with open(path_dataset_src + train_files, 'r') as f: train_files = f.read().splitlines()
with open(path_dataset_src + test_files, 'r') as f: test_files = f.read().splitlines()

add_to_train = []
add_to_test = []

for f in train_files:
    f_layer1 = f.replace('.aedat', '_layer1.mat')
    add_to_train.append(f_layer1)
    f_layer2 = f.replace('.aedat', '_layer2.mat')
    add_to_train.append(f_layer2)

for _ in add_to_train:
    train_files.append(_)

for f in test_files:
    f_layer1 = f.replace('.aedat', '_layer1.mat')
    add_to_test.append(f_layer1)
    f_layer2 = f.replace('.aedat', '_layer2.mat')
    add_to_test.append(f_layer2)

for _ in add_to_test:
    test_files.append(_)
    
# Load and stor dataset event samples
def store_samples(events_files, mode):
    for events_file in tqdm(events_files):
        if events_file == '': continue

        root, extension = os.path.splitext(events_file)
        if extension == '.mat':
            if 'layer1' in root:
                labels = pd.read_csv(path_dataset_src + events_file.replace('_layer1.mat', '_labels.csv'))
                events = SME.spiking_motion_layer1_mat_to_events(path_dataset_src + events_file)   
            elif 'layer2' in root:     
                labels = pd.read_csv(path_dataset_src + events_file.replace('_layer2.mat', '_labels.csv'))
                events = SME.spiking_motion_layer2_mat_to_events(path_dataset_src + events_file)       
        elif extension == '.aedat':
            labels = pd.read_csv(path_dataset_src + events_file.replace('.aedat', '_labels.csv'))
            shape, events = load_events_from_file(path_dataset_src + events_file, parser=aermanager.parsers.parse_dvs_ibm)

        # Load user samples
        # Class 0 for non action
        time_segment_class = []     # [(t_init, t_end, class)]
        prev_event = 0
        for _,row in labels.iterrows():
            if (row.startTime_usec-1 - prev_event) > 0: time_segment_class.append((prev_event, row.startTime_usec-1, 0, (row.startTime_usec-1 - prev_event)/1000))
            if ((row.endTime_usec - row.startTime_usec)) > 0: time_segment_class.append((row.startTime_usec, row.endTime_usec, row['class'], (row.endTime_usec - row.startTime_usec)/1000))
            prev_event = row.endTime_usec + 1 
        time_segment_class.append((prev_event, np.inf, 0))

        total_events = []
        curr_event = []
        # for e in tqdm(events):
        for i, e in enumerate(events):
            if e[2] >= time_segment_class[0][1]:
                # Store event
                if len(curr_event) > 1:total_events.append((time_segment_class[0], len(curr_event), np.array(curr_event)))
                else:
                    if extension == '.mat':
                        print(' ** EVENT ERROR:', events_file.replace('.mat','') + '_num{:02d}_label{:02d}.pckl'.format(i, time_segment_class[0][2]), time_segment_class[0], len(curr_event))
                    elif extension == '.aedat':
                        print(' ** EVENT ERROR:', events_file.replace('.aedat','') + '_num{:02d}_label{:02d}.pckl'.format(i, time_segment_class[0][2]), time_segment_class[0], len(curr_event))
                time_segment_class = time_segment_class[1:]
                curr_event = []
            curr_event.append(list(e))
            
        if len(curr_event) > 1: total_events.append((time_segment_class[0], len(curr_event), np.array(curr_event)))
        time_segment_class = time_segment_class[1:]
        curr_event = []
        
        for i, (meta, _, events) in enumerate(total_events):
            if extension == '.mat':
                
                pickle.dump((events, meta[2]), 
                        open(os.path.join(path_dataset_dst, mode, events_file.replace('.mat','') + '_num{:02d}_label{:02d}.pckl'.format(i, meta[2])), 'wb'))
            elif extension == '.aedat':
                pickle.dump((events, meta[2]), 
                        open(os.path.join(path_dataset_dst, mode, events_file.replace('.aedat','') + '_num{:02d}_label{:02d}.pckl'.format(i, meta[2])), 'wb'))
        


store_samples(train_files, 'train')
store_samples(test_files, 'test')

