import os
import pandas as pd
import numpy as np
import sparse
import pickle

import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import scipy.io
import my_nda_aug


np.random.seed(0)

chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height, width = 260, 346
size = 0.25


# Source data folder
path_dataset = '../datasets/FDD_SME_nda/'
files = os.listdir(path_dataset + 'all_aedat4_mat/')
parser = aermanager.parsers.parse_aedat4                                                                                                           


# Used for .mat from spiking_motion_layer1 event output
def spiking_motion_layer1_mat_to_events(filename):
    f = scipy.io.loadmat(filename, mat_dtype=True)

    layer1_x = f['layer1_output']['x']
    layer1_y = f['layer1_output']['y']
    layer1_ts = f['layer1_output']['ts']
    layer1_sp = f['layer1_output']['sp']
    layer1_dir = f['layer1_output']['dir']
    layer1_p = f['layer1_output']['p']
    
    x = layer1_x[0][0].astype('uint64').reshape(layer1_x[0][0].size,)
    y = layer1_y[0][0].astype('uint64').reshape(layer1_y[0][0].size,)
    t = layer1_ts[0][0].astype('uint64').reshape(layer1_ts[0][0].size,)
    sp = layer1_sp[0][0].astype('uint64').reshape(layer1_sp[0][0].size,)
    dir = layer1_dir[0][0].astype('uint64').reshape(layer1_dir[0][0].size,)
    p = layer1_p[0][0].astype('uint64').reshape(layer1_p[0][0].size,)
    
    events_struct = [("x", np.uint64), ("y", np.uint64), ("t", np.uint64), ("p", np.uint64), ("sp", np.uint64), ("dir", np.uint64) ]
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

    x = layer2_x[0][0].astype('uint64').reshape(layer2_x[0][0].size,)
    y = layer2_y[0][0].astype('uint64').reshape(layer2_y[0][0].size,)
    t = layer2_ts[0][0].astype('uint64').reshape(layer2_ts[0][0].size,)
    sp = layer2_sp[0][0].astype('uint64').reshape(layer2_sp[0][0].size,)
    dir = layer2_dir[0][0].astype('uint64').reshape(layer2_dir[0][0].size,)
    p = layer2_p[0][0].astype('uint64').reshape(layer2_p[0][0].size,)

    events_struct = [("x", np.uint64), ("y", np.uint64), ("t", np.uint64), ("p", np.uint64), ("sp", np.uint64), ("dir", np.uint64) ]
    motion_event = np.fromiter(zip(x, y, t, p, sp, dir), dtype=events_struct)
    return motion_event

def find_closest_value(motion_events_value, row_value):
    # 执行二分查找，找到插入点索引
    insert_index = np.searchsorted(motion_events_value, row_value)

    # 确定最接近值的索引
    if insert_index == 0:
        closest_index = 0
    elif insert_index == len(motion_events_value):
        closest_index = len(motion_events_value) - 1
    else:
        # 比较插入点前后的值，选择更接近的
        before_value = motion_events_value[insert_index - 1]
        after_value = motion_events_value[insert_index]
        if abs(before_value - row_value) < abs(after_value - row_value):
            closest_index = insert_index - 1
        else:
            closest_index = insert_index

    # 获取最接近值及其相关信息
    closest_value = motion_events_value[closest_index]
    # 可以根据需要提取其他相关信息，如 motion_events 中的其他字段
    # print("最接近的值:", closest_value)

    return closest_value



# Target data folder
if not os.path.isdir(path_dataset + 'FDD_SME_splits'):
    os.mkdir(path_dataset + 'FDD_SME_splits')
    os.makedirs(path_dataset + 'FDD_SME_splits/' + f'dataset_4sets_{chunk_len_us}/train')
    os.makedirs(path_dataset + 'FDD_SME_splits/' + f'dataset_4sets_{chunk_len_us}/test')
 
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


# Find aedat_files in train samples and test samples
train_samples_4sets_aedat_files = [_ for _ in train_samples_4sets if _.endswith('.aedat4')]
test_samples_4sets_aedat_files = [_ for _ in test_samples_4sets if _.endswith('.aedat4')]

# Add spiking motion files into train samples list
for f in train_samples_4sets_aedat_files:
    f_layer1 = f.replace('.aedat4', '_layer1.mat')
    train_samples_4sets.append(f_layer1)
    f_layer2 = f.replace('.aedat4', '_layer2.mat')
    train_samples_4sets.append(f_layer2)

no_aug_4sets = []
# Avoid adding spiking motion files into test samples list
for f in test_samples_4sets_aedat_files:
    f_layer1 = f.replace('.aedat4', '_layer1.mat')
    no_aug_4sets.append(f_layer1)
    f_layer2 = f.replace('.aedat4', '_layer2.mat')
    no_aug_4sets.append(f_layer2)



for events_file in tqdm(files):
    root, extension = os.path.splitext(events_file)
    if extension == '.mat':
        if events_file[-11:]=='_layer1.mat':
            labels = pd.read_csv(path_dataset + 'labels/'+ events_file.replace('_layer1.mat', '.csv'))
            motion_events = spiking_motion_layer1_mat_to_events(path_dataset + 'all_aedat4_mat/' + events_file)
            filename_dst = path_dataset + 'FDD_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
            events_file.replace('.mat', '_{}_{}.pckl')  
        elif events_file[-11:]=='_layer2.mat':     
            labels = pd.read_csv(path_dataset + 'labels/'+ events_file.replace('_layer2.mat', '.csv'))
            motion_events = spiking_motion_layer2_mat_to_events(path_dataset + 'all_aedat4_mat/' + events_file)
            filename_dst = path_dataset + 'FDD_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
            events_file.replace('.mat', '_{}_{}.pckl')
    elif extension == '.aedat4':
        shape, events = load_events_from_file(path_dataset + 'all_aedat4_mat/' + events_file, parser=parser)
        labels = pd.read_csv(path_dataset + 'labels/' + events_file.replace('.aedat4', '.csv'))
        filename_dst = path_dataset + 'FDD_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
            events_file.replace('.aedat4', '_{}_{}.pckl')

    for _,row in labels.iterrows():
    
        print('events_file = ', events_file)

        if extension == '.aedat4':
            startTime_event_num = np.where(events['t']==row.startTime_ev)[0][0]
            endTime_event_num = np.where(events['t']==row.endTime_ev)[0][0]
            
            sample_events = events[startTime_event_num:endTime_event_num]
            total_events = np.array([sample_events['x'], sample_events['y'], sample_events['t'], sample_events['p']]).transpose()
            breakpoint()
        elif extension == '.mat':
            start_closest_value = find_closest_value(motion_events['t'], row.startTime_ev)
            end_closest_value = find_closest_value(motion_events['t'], row.endTime_ev)
            startTime_event_num = np.where(motion_events['t']==start_closest_value)[0][0]
            endTime_event_num = np.where(motion_events['t']==end_closest_value)[0][0]
            
            sample_events = motion_events[startTime_event_num:endTime_event_num]
            total_events = np.array([sample_events['x'], sample_events['y'], sample_events['t'], sample_events['p'], sample_events['sp'], sample_events['dir']]).transpose()
            breakpoint()

        total_chunks = []
        
        while total_events.shape[0] > 0:
            end_t = total_events[-1][2]
            chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
            
            if total_events.shape[1] == 4:
                if len(chunk_inds) <= 4: 
                    pass
                else:
                    total_chunks.append(total_events[chunk_inds])
            elif total_events.shape[1] == 6:
                if len(chunk_inds) <= 0: 
                    pass
                else:
                    total_chunks.append(total_events[chunk_inds])
            
            total_events = total_events[:max(1, chunk_inds.min())-1]
        
        if len(total_chunks) == 0:  continue
        total_chunks = total_chunks[::-1]
        
        total_frames = []
        total_speed = []
        total_direction = []

        for chunk in total_chunks:
            frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                               np.ones(chunk.shape[0]).astype('int32'), 
                               (width, height, 2))   # .to_dense()
            total_frames.append(frame)
            
            if(chunk.shape[1] == 6):
                frame_speed = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                                   chunk[:,4].astype('int32'), 
                                   (width, height, 2))   # .todense()
                total_speed.append(frame_speed)
                

                frame_direction = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                                   chunk[:,5].astype('int32'), 
                                   (width, height, 2))   # .todense()
                total_direction.append(frame_direction)

        total_frames = sparse.stack(total_frames)        
        total_frames = np.clip(total_frames, a_min=0, a_max=255)
        total_frames = total_frames.astype('uint8')

        if total_speed != []:
            total_speed = sparse.stack(total_speed)
            total_speed = np.clip(total_speed, a_min=0, a_max=255)
            total_speed = total_speed.astype('uint8')          

        if total_direction != []:
            total_direction = sparse.stack(total_direction)
            total_direction = np.clip(total_direction, a_min=0, a_max=255)
            total_direction = total_direction.astype('uint8')      

        

        if   '_4' or '-4' in events_file:  val_set = 'S4' 
        elif '_3' or '-3' in events_file:  val_set = 'S3' 
        elif '_2' or '-2' in events_file:  val_set = 'S2' 
        elif '_1' or '-1' in events_file:  val_set = 'S1'
        else: raise ValueError('Set not handled')
        
        
        
        if events_file in train_samples_4sets:
            
            filename_dst_final = filename_dst.format('4sets', 'train', val_set, row['class'])
            
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))
            print("Saved file: ", filename_dst_final)

            aug_total_frames = my_nda_aug.my_nda_aug(total_frames, 'roll')
            aug_filename_dst = filename_dst_final[:104] + 'aug_roll_' + filename_dst_final[104:]
            pickle.dump(aug_total_frames, open(aug_filename_dst, 'wb'))
            print("Saved file: ", aug_filename_dst)

            aug_total_frames = my_nda_aug.my_nda_aug(total_frames, 'rotate')
            aug_filename_dst = filename_dst_final[:104] + 'aug_rotate_' + filename_dst_final[104:]
            pickle.dump(aug_total_frames, open(aug_filename_dst, 'wb'))
            print("Saved file: ", aug_filename_dst)

            aug_total_frames = my_nda_aug.my_nda_aug(total_frames, 'shear')
            aug_filename_dst = filename_dst_final[:104] + 'aug_shear_' + filename_dst_final[104:]
            pickle.dump(aug_total_frames, open(aug_filename_dst, 'wb'))
            print("Saved file: ", aug_filename_dst)

            if(isinstance(total_speed, sparse.COO)):
                speed_filename = (filename_dst_final[:104]+"speed_"+filename_dst_final[104:])
                pickle.dump(total_speed, open(speed_filename, 'wb'))
                print("Saved file: ", speed_filename)

            if(isinstance(total_direction, sparse.COO)):
                direction_filename = (filename_dst_final[:104]+"direction_"+filename_dst_final[104:])
                pickle.dump(total_direction, open(direction_filename, 'wb'))
                print("Saved file: ", speed_filename)

        if events_file in test_samples_4sets:
            
            filename_dst_final = filename_dst.format('4sets', 'test', val_set, row['class'])
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))
            print("Saved file: ", filename_dst_final)
