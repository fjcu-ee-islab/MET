import os
import pandas as pd
import numpy as np
import sparse
import pickle
import glob
import h5py

import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import scipy.io
import SME

mapping = { 0 :'Basketball',
            1 :'Biking',
            2 :'Diving',
            3 :'GolfSwing',
            4 :'HorseRiding',
            5 :'SoccerJuggling',
            6 :'Swing',
            7 :'TennisSwing',
            8 :'TrampoloneJumping',
            9 :'VolleyballSpiking',
            10 :'WalkingWithDog'}
            
def UCF11_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("UCF11 Dataset not found, looked at: {}".format(dataset_path))

    train_files = []
    test_files = []
    for digit in range(11):
        digit_txt_train = glob.glob(os.path.join(dataset_path, 'Train/{}/*.txt'.format(digit)))
        digit_txt_test = glob.glob(os.path.join(dataset_path, 'Test/{}/*.txt'.format(digit))) 

        train_files.append(digit_txt_train)
        test_files.append(digit_txt_test)
        
    # We need the same number of train and test samples for each digit, let's compute the minimum
    max_n_train = min(map(lambda l: len(l), train_files))
    max_n_test = min(map(lambda l: len(l), test_files))
    n_train = max_n_train # we could take max_n_train, but my memory on the shared drive is full
    n_test = max_n_test # we test on the whole test set - lets only take 100*10 samples
    assert((n_train <= max_n_train) and (n_test <= max_n_test)), 'Requested more samples than present in dataset'

    print("UCF11: {} train samples and {} test samples per digit (max: {} train and {} test)".format(n_train, n_test, max_n_train, max_n_test))
    # Crop extra samples of each digits
    train_files = map(lambda l: l[:n_train], train_files)
    test_files = map(lambda l: l[:n_test], test_files)

    return list(train_files), list(test_files)

def UCF11_load_events_from_txt(file_path, max_duration=None):
    data = np.loadtxt(file_path)
    return data


np.random.seed(0)

chunk_len_ms = 96
chunk_len_us = chunk_len_ms*1000
height, width = 240, 240
size = 0.25


# Source data folder
OF_dir = '../datasets/UCF11_OF/UCF11_total_flow/'
path_dataset = '../datasets/UCF11/'
Output_dir = '../datasets/UCF11_OF/'

fns_train, fns_test = UCF11_get_file_names(path_dataset+'/txt')
fns_train = [val for sublist in fns_train for val in sublist]
fns_test = [val for sublist in fns_test for val in sublist]
files = fns_train+fns_test
train_label_list = [[] for i in range(11)]
test_label_list = [[] for i in range(11)]

total_chunk_time_window_length = 0
total_chunk_time_window_length_total = 0
total_chunk_num = 0

for events_file in tqdm(files):
    if events_file in fns_train: continue
    print('events_file = ', events_file)

    root, extension = os.path.splitext(events_file)
         
    if extension == '.txt':
        data = UCF11_load_events_from_txt(events_file)
        times = data[:,0] # [ts]
        addrs = data[:,1:] # [p, x, y] 
        label = int(events_file.split('/')[-2])
        filename_dst = Output_dir + 'UCF11_OF_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
            events_file.split('/')[-1].replace('.txt', '_{}.pckl')
    
    if extension == '.txt':
        total_events = np.array([addrs[:,1], addrs[:,2], times, addrs[:,0]]).T
    
    classes = events_file.split('/')[-2]
    file_name = events_file.split('/')[-1].split('.')[0]
    OF_file = OF_dir + classes + '/total_flow_' + file_name + '.h5'

    if classes == '7' and file_name == '00104':
        continue
    elif classes == '8' and file_name == '00023':
        continue
    elif classes == '10' and file_name == '00090':
        continue
    else:
        with h5py.File(OF_file, 'r') as hf:
            ts_array = np.array(list(hf.keys())).astype('int')
            #breakpoint()
            startTime = int(times.min())
            endTime = int(times.max())
            ts_array_index_in_range = np.where((ts_array>startTime) & (ts_array<endTime))
            ts_array_in_range = ts_array[ts_array_index_in_range]
            
            # 計算所有keys對應的數據的總數量
            total_length = sum(len(hf[str(ts)][()]) for ts in ts_array_in_range)

            # 預先分配一個大數組來存儲所有數據
            final_array = np.empty(total_length, dtype=[('x', '<u4'), ('y', '<u4'), ('t', '<u4'), ('u', '<f4'), ('v', '<f4')])

            # 初始化索引變量來跟踪數據應插入的位置
            start_idx = 0
            length_array = []
            
            # 遍歷ts_array_in_range中的每個key
            for ts in ts_array_in_range:
                key = str(ts)
                
                # 讀取該key下存儲的數據
                data = hf[key][()]
                
                # 計算數據的長度
                length = len(data)
                length_array.append(length)
                
                # 將數據插入到預先分配的大數組中
                final_array[start_idx:start_idx + length] = data
                
                # 更新索引變量
                start_idx += length
                                    
            OF_events = np.array([final_array['x'], final_array['y'], final_array['t'], np.full(final_array['u'].shape, 1), np.around(final_array['u']), np.around(final_array['v'])]).astype(np.int64).transpose()

    total_chunks = []
    OF_total_chunks = []

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
                
                time_window_length = total_events[chunk_inds[-1]][2] - total_events[chunk_inds[0]][2]
                total_chunk_time_window_length_total = total_chunk_time_window_length_total + time_window_length
                total_chunk_num += 1
                   
        # Event data with speed and direction (.mat)

        total_events = total_events[:max(1, chunk_inds.min())-1]

    # Every chunk_len_us
    while OF_events.shape[0] > 0:
        end_t = OF_events[-1][2]
        OF_chunk_inds = np.where(OF_events[:,2] >= end_t - chunk_len_us)[0]
        
        # If the numbers of the last events of chunk_len_us (default: 12000 us) <= 4 
        # Only event data (.aedat)
        if OF_events.shape[1] == 6:
            if len(OF_chunk_inds) <= 4: 
                pass
            else:
                OF_total_chunks.append(OF_events[OF_chunk_inds])
                
                time_window_length = OF_events[OF_chunk_inds[-1]][2] - OF_events[OF_chunk_inds[0]][2]
                total_chunk_time_window_length_total = total_chunk_time_window_length_total + time_window_length
                total_chunk_num += 1
                    
        else: breakpoint()

        # Remaining events.
        OF_events = OF_events[:max(1, OF_chunk_inds.min())-1]

    if len(total_chunks) == 0: 
        continue
    total_chunks = total_chunks[::-1]
    if len(OF_total_chunks) == 0: 
        continue
    OF_total_chunks = OF_total_chunks[::-1]    
    
    total_frames = []
    OF_total_frames_u, OF_total_frames_v = [], []   
    
    for chunk in total_chunks:
        frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                            np.ones(chunk.shape[0]).astype('int32'), 
                            (width, height, 2))   # .to_dense()
        total_frames.append(frame)

    for OF_chunk in OF_total_chunks:    

        # u
        OF_frame_u = sparse.COO(OF_chunk[:,[0,1,3]].transpose().astype('int32'), 
                            OF_chunk[:,4].astype('int32'), 
                            (height, width, 2))   # .todense()
        OF_total_frames_u.append(OF_frame_u)

        # v
        OF_frame_v = sparse.COO(OF_chunk[:,[0,1,3]].transpose().astype('int32'), 
                            OF_chunk[:,5].astype('int32'), 
                            (height, width, 2))   # .todense()
        OF_total_frames_v.append(OF_frame_v)          
            
    total_frames = sparse.stack(total_frames)
    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')

    OF_total_frames_u = sparse.stack(OF_total_frames_u)
    OF_total_frames_u = np.clip(OF_total_frames_u, a_min=0, a_max=255)
    OF_total_frames_u = OF_total_frames_u.astype('uint8')  

    OF_total_frames_v = sparse.stack(OF_total_frames_v)
    OF_total_frames_v = np.clip(OF_total_frames_v, a_min=0, a_max=255)
    OF_total_frames_v = OF_total_frames_v.astype('uint8')  


    if events_file in fns_train:
        train_file_dst = (Output_dir + 'UCF11_OF_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/').format('11sets','train')
        
        os.makedirs(train_file_dst, exist_ok=True)
        train_file_name = filename_dst.format('11sets','train', label)
        pickle.dump(total_frames, open(train_file_name, 'wb'))
        pickle.dump(OF_total_frames_u, open(train_file_name[:64]+ 'OF_u_' + train_file_name[64:], 'wb'))
        pickle.dump(OF_total_frames_v, open(train_file_name[:64]+ 'OF_v_' + train_file_name[64:], 'wb'))

        aug_roll_total_frames = SME.nda(total_frames, 'roll')
        aug_roll_total_frames_dst = train_file_name[:64] + 'aug_roll_' + train_file_name[64:]
        if os.path.isfile(aug_roll_total_frames_dst): 
            print(aug_roll_total_frames_dst, "exists.")
            pass
        else: pickle.dump(aug_roll_total_frames, open(aug_roll_total_frames_dst, 'wb'))

        aug_rotate_total_frames = SME.nda(total_frames, 'rotate')
        aug_rotate_total_frames_dst = train_file_name[:64] + 'aug_rotate_' + train_file_name[64:]
        if os.path.isfile(aug_rotate_total_frames_dst): 
            print(aug_rotate_total_frames_dst, "exists.")
            pass
        else: pickle.dump(aug_rotate_total_frames, open(aug_rotate_total_frames_dst, 'wb'))

        aug_shear_total_frames = SME.nda(total_frames, 'shear')
        aug_shear_total_frames_dst = train_file_name[:64] + 'aug_shear_' + train_file_name[64:]
        if os.path.isfile(aug_shear_total_frames_dst): 
            print(aug_shear_total_frames_dst, "exists.")
            pass
        else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst, 'wb'))

        #u
        aug_roll_total_frames_u = SME.nda(OF_total_frames_u, 'roll')
        aug_roll_total_frames_dst_u = train_file_name[:64] + 'aug_roll_' + train_file_name[64:]
        if os.path.isfile(aug_roll_total_frames_dst_u): 
            print(aug_roll_total_frames_dst_u, "exists.")
            pass
        else: pickle.dump(aug_roll_total_frames_u, open(aug_roll_total_frames_dst_u, 'wb'))

        aug_rotate_total_frames_u = SME.nda(OF_total_frames_u, 'rotate')
        aug_rotate_total_frames_dst_u = train_file_name[:64] + 'aug_rotate_' + train_file_name[64:]
        if os.path.isfile(aug_rotate_total_frames_dst_u): 
            print(aug_rotate_total_frames_dst_u, "exists.")
            pass
        else: pickle.dump(aug_rotate_total_frames_u, open(aug_rotate_total_frames_dst_u, 'wb'))

        aug_shear_total_frames_u = SME.nda(OF_total_frames_u, 'shear')
        aug_shear_total_frames_dst_u = train_file_name[:64] + 'aug_shear_' + train_file_name[64:]
        if os.path.isfile(aug_shear_total_frames_dst_u): 
            print(aug_shear_total_frames_dst_u, "exists.")
            pass
        else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst_u, 'wb'))

        #v
        aug_roll_total_frames_v = SME.nda(OF_total_frames_v, 'roll')
        aug_roll_total_frames_dst_v = train_file_name[:64] + 'aug_roll_' + train_file_name[64:]
        if os.path.isfile(aug_roll_total_frames_dst_v): 
            print(aug_roll_total_frames_dst_v, "exists.")
            pass
        else: pickle.dump(aug_roll_total_frames_v, open(aug_roll_total_frames_dst_v, 'wb'))

        aug_rotate_total_frames_v = SME.nda(OF_total_frames_v, 'rotate')
        aug_rotate_total_frames_dst_v = train_file_name[:64] + 'aug_rotate_' + train_file_name[64:]
        if os.path.isfile(aug_rotate_total_frames_dst_v): 
            print(aug_rotate_total_frames_dst_v, "exists.")
            pass
        else: pickle.dump(aug_rotate_total_frames_v, open(aug_rotate_total_frames_dst_v, 'wb'))

        aug_shear_total_frames_v = SME.nda(OF_total_frames_v, 'shear')
        aug_shear_total_frames_dst_v = train_file_name[:64] + 'aug_shear_' + train_file_name[64:]
        if os.path.isfile(aug_shear_total_frames_dst_v): 
            print(aug_shear_total_frames_dst_v, "exists.")
            pass
        else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst_v, 'wb'))

    if events_file in fns_test:
        test_file_dst = (Output_dir + 'UCF11_SME_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/').format('11sets','test')
        os.makedirs(test_file_dst, exist_ok=True)
        test_file_name = filename_dst.format('11sets','test', label)
        if os.path.isfile(test_file_name): 
            print(test_file_name, "exists.")
            pass
        else: pickle.dump(total_frames, open(test_file_name, 'wb'))
        
print("total_chunk_num = ", total_chunk_num)
print("total_chunk_time_window_length_total = ", total_chunk_time_window_length_total)
print("Average time window length = ", total_chunk_time_window_length_total / total_chunk_num/1000, "ms")
