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
import h5py
import csv
import SME

np.random.seed(0)

chunk_len_ms = 72
chunk_len_us = chunk_len_ms*1000
height, width = 128, 128
size = 0.25

# Source data folder
OF_dir = '../datasets/SL_Animals_OF/SL_Animals_total_flow/'
events_file_dir = '../datasets/SL_Animals_OF/allusers_aedat/'
path_dataset = '../datasets/SL_Animals_OF/'
files = os.listdir(path_dataset + 'allusers_aedat/')

parser = aermanager.parsers.parse_dvs_128

# Target data folder
if not os.path.isdir(path_dataset + 'SL_animal_splits'):
    os.mkdir(path_dataset + 'SL_animal_splits')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_4sets_{chunk_len_us}/train')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_4sets_{chunk_len_us}/test')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_3sets_{chunk_len_us}/train')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_3sets_{chunk_len_us}/test')

test_samples_4sets = [ 'user10_indoor.aedat', 'user12_indoor.aedat', 'user14_indoor.aedat', 'user17_indoor.aedat', 'user19_sunlight.aedat', 
                      'user24_sunlight.aedat', 'user29_imse.aedat', 'user30_imse.aedat', 'user34_imse.aedat', 'user35_dc.aedat', 
                      'user36_dc.aedat', 'user37_dc.aedat', 'user38_dc.aedat', 'user42_dc.aedat', 'user57_dc.aedat' ]
train_samples_4sets = [ f for f in files if f not in test_samples_4sets ]

test_samples_3sets = [ 'user10_indoor.aedat', 'user12_indoor.aedat', 'user14_indoor.aedat', 'user17_indoor.aedat', 'user19_sunlight.aedat', 
                      'user24_sunlight.aedat', 'user29_imse.aedat', 'user30_imse.aedat', 'user34_imse.aedat']
train_samples_3sets = [ f for f in files if f not in test_samples_4sets and '_dc' not in f ]                                      

for events_file in tqdm(files):
    if events_file == '': continue
    root, extension = os.path.splitext(events_file) 
                 
    if extension == '.aedat':
        shape, events = load_events_from_file(events_file_dir + events_file, parser=parser)
        labels = pd.read_csv(path_dataset + 'tags_updated_19_08_2020/' + events_file.replace('.aedat', '.csv'))
        filename_dst = path_dataset + 'SL_animal_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
        events_file.replace('.aedat', '_{}_{}.pckl')

    else: continue
    
    

    for _,row in labels.iterrows():
        
        if extension == '.aedat':
            sample_events = events[row.startTime_ev:row.endTime_ev]
            total_events = np.array([sample_events['x'], sample_events['y'], sample_events['t'], sample_events['p']]).transpose()
            
        OF_file = OF_dir + 'total_flow_' + root + '.h5'
        #print("Opening ", OF_file)

        with h5py.File(OF_file, 'r') as hf:
            ts_array = np.array(list(hf.keys())).astype('int')
            startTime = int(events[row.startTime_ev][2])
            endTime = int(events[row.endTime_ev][2])
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

        # Every chunk_len_us
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
            else: breakpoint()
            
            # Remaining events.
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
            else: breakpoint()

            # Remaining events.
            OF_events = OF_events[:max(1, OF_chunk_inds.min())-1]
        

        if len(total_chunks) == 0: 
            breakpoint()
            continue
        total_chunks = total_chunks[::-1]
        if len(OF_total_chunks) == 0: 
            breakpoint()
            continue
        OF_total_chunks = OF_total_chunks[::-1]            

        total_frames, OF_total_frames_u_pos, OF_total_frames_u_neg, OF_total_frames_v_pos, OF_total_frames_v_neg = [], [], [], [], []
        
        for chunk in total_chunks:    
            
            frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                                np.ones(chunk.shape[0]).astype('int32'), 
                                (height, width, 2))   # .todense()
            total_frames.append(frame)


        for OF_chunk in OF_total_chunks:    
            
            # u            
            # 使用布林索引來找到正值和負值
            OF_chunk_positive = OF_chunk[OF_chunk[:, 4] > 0]
            OF_chunk_negative = OF_chunk[OF_chunk[:, 4] < 0]
            OF_chunk_negative[:, 4] = np.abs(OF_chunk_negative[:, 4])
            OF_chunk_negative[:, 5] = np.abs(OF_chunk_negative[:, 5])

            OF_frame_u_pos = sparse.COO(OF_chunk[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk[:,4].astype('int32'), 
                                (height, width, 2))   # .todense()
            OF_total_frames_u_pos.append(OF_frame_u_pos)
            OF_frame_u_neg = sparse.COO(OF_chunk[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk[:,4].astype('int32'), 
                                (height, width, 2))   # .todense()
            OF_total_frames_u_neg.append(OF_frame_u_neg)

            # v
            OF_frame_v_pos = sparse.COO(OF_chunk[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk[:,5].astype('int32'), 
                                (height, width, 2))   # .todense()
            OF_total_frames_v_pos.append(OF_frame_v_pos)   
            OF_frame_v_neg = sparse.COO(OF_chunk[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk[:,5].astype('int32'), 
                                (height, width, 2))   # .todense()
            OF_total_frames_v_neg.append(OF_frame_v_neg)                             
                        
        
        total_frames = sparse.stack(total_frames)
        total_frames = np.clip(total_frames, a_min=0, a_max=255)
        total_frames = total_frames.astype('uint8')    

        OF_total_frames_u_pos = sparse.stack(OF_total_frames_u_pos)
        OF_total_frames_u_pos = np.clip(OF_total_frames_u_pos, a_min=0, a_max=255)
        OF_total_frames_u_pos = OF_total_frames_u_pos.astype('uint8')  

        OF_total_frames_u_neg = sparse.stack(OF_total_frames_u_neg)
        OF_total_frames_u_neg = np.clip(OF_total_frames_u_neg, a_min=0, a_max=255)
        OF_total_frames_u_neg = OF_total_frames_u_neg.astype('uint8')  

        OF_total_frames_v_pos = sparse.stack(OF_total_frames_v_pos)
        OF_total_frames_v_pos = np.clip(OF_total_frames_v_pos, a_min=0, a_max=255)
        OF_total_frames_v_pos = OF_total_frames_v_pos.astype('uint8')  

        OF_total_frames_v_neg = sparse.stack(OF_total_frames_v_neg)
        OF_total_frames_v_neg = np.clip(OF_total_frames_v_neg, a_min=0, a_max=255)
        OF_total_frames_v_neg = OF_total_frames_v_neg.astype('uint8')  

        if '_sunlight' in events_file:  val_set = 'S4' # S4 indoors with frontal sunlight
        elif '_indoor' in events_file:  val_set = 'S3' # S3 indoors neon light
        elif '_dc' in events_file:      val_set = 'S2' # S2 natural side light
        elif '_imse' in events_file:    val_set = 'S1' # S1 natural side light
        else: raise ValueError('Set not handled')
        
        '''
        if events_file in train_samples_3sets:     
            filename_dst_final = filename_dst.format('3sets', 'train', val_set, row['class'])
               
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))            
            pickle.dump(OF_total_frames_u_pos, open(filename_dst_final[:69] + 'OF_u_pos_' + filename_dst_final[69:], 'wb'))
            pickle.dump(OF_total_frames_u_neg, open(filename_dst_final[:69] + 'OF_u_neg_' + filename_dst_final[69:], 'wb'))                        
            pickle.dump(OF_total_frames_v_pos, open(filename_dst_final[:69] + 'OF_v_pos_' + filename_dst_final[69:], 'wb'))
            pickle.dump(OF_total_frames_u_neg, open(filename_dst_final[:69] + 'OF_u_neg_' + filename_dst_final[69:], 'wb'))  
           
            # Augmentation to whole total_frames
            aug_roll_total_frames = SME.nda(total_frames, 'roll')
            pickle.dump(aug_roll_total_frames, open(filename_dst_final[:69] + 'aug_roll_' + filename_dst_final[69:], 'wb'))
            aug_rotate_total_frames = SME.nda(total_frames, 'rotate')
            pickle.dump(aug_rotate_total_frames, open(filename_dst_final[:69] + 'aug_rotate_' + filename_dst_final[69:], 'wb'))
            aug_shear_total_frames = SME.nda(total_frames, 'shear')
            pickle.dump(aug_shear_total_frames, open(filename_dst_final[:69] + 'aug_shear_' + filename_dst_final[69:], 'wb'))
        '''
            
        
        if events_file in train_samples_4sets:
            filename_dst_final = filename_dst.format('4sets', 'train', val_set, row['class'])
            
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))            
            pickle.dump(OF_total_frames_u_pos, open(filename_dst_final[:69] + 'OF_u_pos_' + filename_dst_final[69:], 'wb'))
            pickle.dump(OF_total_frames_u_neg, open(filename_dst_final[:69] + 'OF_u_neg_' + filename_dst_final[69:], 'wb'))                        
            pickle.dump(OF_total_frames_v_pos, open(filename_dst_final[:69] + 'OF_v_pos_' + filename_dst_final[69:], 'wb'))
            pickle.dump(OF_total_frames_v_neg, open(filename_dst_final[:69] + 'OF_v_neg_' + filename_dst_final[69:], 'wb'))  
           
            # Augmentation to whole total_frames
            aug_roll_total_frames = SME.nda(total_frames, 'roll')
            pickle.dump(aug_roll_total_frames, open(filename_dst_final[:69] + 'aug_roll_' + filename_dst_final[69:], 'wb'))
            aug_rotate_total_frames = SME.nda(total_frames, 'rotate')
            pickle.dump(aug_rotate_total_frames, open(filename_dst_final[:69] + 'aug_rotate_' + filename_dst_final[69:], 'wb'))
            aug_shear_total_frames = SME.nda(total_frames, 'shear')
            pickle.dump(aug_shear_total_frames, open(filename_dst_final[:69] + 'aug_shear_' + filename_dst_final[69:], 'wb'))
            


            
            
        '''
        if events_file in test_samples_3sets:
            filename_dst_final = filename_dst.format('3sets', 'test', val_set, row['class'])              
            
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))
        '''
                            
        if events_file in test_samples_4sets:       
            filename_dst_final = filename_dst.format('4sets', 'test', val_set, row['class'])

            pickle.dump(total_frames, open(filename_dst_final, 'wb'))
