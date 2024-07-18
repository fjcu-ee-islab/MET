import os
import pandas as pd
import numpy as np
import sparse
import pickle
import h5py

import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import scipy.io
import SME

np.random.seed(0)

chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height, width = 260, 346
size = 0.25


# Source data folder
Original_aedat4_dir = '../datasets/FDD_OF/AEDAT4.0/'
parser = aermanager.parsers.parse_aedat4  
Labels_dir = '../datasets/FDD_OF/labels/'

OF_dir = '../datasets/FDD_OF/FDD_total_flow/'
path_dataset = '../datasets/FDD_OF/OF_input/test/'
Output_dir = '../datasets/FDD_OF/'
event_files = os.listdir(path_dataset)
                                                                                                      


# Target data folder

if not os.path.isdir(Output_dir + 'FDD_OF_splits'):
    os.mkdir(Output_dir + 'FDD_OF_splits')
    os.makedirs(Output_dir + 'FDD_OF_splits/' + f'dataset_4sets_{chunk_len_us}/train')
    os.makedirs(Output_dir + 'FDD_OF_splits/' + f'dataset_4sets_{chunk_len_us}/test')
 
# Init train samples and test samples
train_samples_4sets, test_samples_4sets = [], []

# Pre-defined train samples list and test samples list
test_4sets_list = Output_dir + 'FDD_OF_splits/' + 'test_4sets_list.txt'
train_4sets_list = Output_dir + 'FDD_OF_splits/' + 'train_4sets_list.txt'


# If samples are given:
f = open(train_4sets_list, 'r')
for index in f:
    train_samples_4sets.append(index[:-1])
f.close()
f = open(test_4sets_list, 'r')
for index in f:
    test_samples_4sets.append(index[:-1])
f.close()



for events_file in tqdm(event_files):
    
    Original_shape, Original_events = load_events_from_file(Original_aedat4_dir + events_file + '.aedat4', parser=parser)
    ts_offset = Original_events[0][2]
    
    labels = pd.read_csv(Labels_dir + events_file + '.csv')

    events_file_dir = path_dataset + events_file + '/events_left/events.h5'
    with h5py.File(events_file_dir, "r") as f: 
        events_struct = np.dtype([("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("p", np.float32)])
        num_events = f['events']['t'].shape[0]
        events_loaded = np.zeros(num_events, dtype=events_struct)
        events_loaded['x'] = f['events']['x']
        events_loaded['y'] = f['events']['y']
        events_loaded['t'] = f['events']['t']
        events_loaded['p'] = f['events']['p']
        
        events = np.array([events_loaded['x'], events_loaded['y'], events_loaded['t'], events_loaded['p']]).astype(np.int64).transpose()            

        print("")

    filename_dst = Output_dir + 'FDD_OF_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
        events_file + '_{}_{}.pckl'

    OF_file = OF_dir + '/total_flow_' + events_file + '.h5'

    with h5py.File(OF_file, 'r') as hf:
        ts_array = np.array(list(hf.keys())).astype('int')
        #ts_offset = ts_array[0]
        #ts_array = ts_array - ts_offset
        
        startTime = int(events_loaded['t'].min())
        endTime = int(events_loaded['t'].max())
        
        ts_array_index_in_range = np.where((ts_array>startTime) & (ts_array<endTime))
        ts_array_in_range = ts_array[ts_array_index_in_range]
        #ts_array_in_range = ts_array_in_range + ts_offset
        
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
                            
        OF_events_whole = np.array([final_array['y'], final_array['x'], final_array['t'], np.full(final_array['u'].shape, 1), np.around(final_array['u']), np.around(final_array['v'])]).astype(np.int64).transpose()
    
    for _,row in labels.iterrows():
        Offset_startTime_ev = row.startTime_ev-ts_offset
        Offset_endTime_ev = row.endTime_ev-ts_offset
        
        # aedat4 events
        startTime_event_num = np.where(events[:, 2]==Offset_startTime_ev)[0][0]
        endTime_event_num = np.where(events[:, 2]==Offset_endTime_ev)[0][0]
        total_events = events[startTime_event_num:endTime_event_num]
        if total_events.shape[0]==0:    breakpoint()
        # OF events        
        OF_startTime_event_num = np.where(OF_events_whole[:, 2]>=Offset_startTime_ev)[0][0]
        OF_endTime_event_num = np.where(OF_events_whole[:, 2]<=Offset_endTime_ev)[0][-1]
        
        OF_events = OF_events_whole[OF_startTime_event_num:OF_endTime_event_num]
        if OF_events.shape[0]==0:    breakpoint()
        total_chunks = []
        OF_total_chunks = []

        while total_events.shape[0] > 0:
            end_t = total_events[-1][2]
            chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
            
            if total_events.shape[1] == 4:
                if len(chunk_inds) <= 4: 
                    pass
                else:
                    total_chunks.append(total_events[chunk_inds])

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
        
        total_frames = []
     

        OF_total_frames_u_pos, OF_total_frames_u_neg, OF_total_frames_v_pos, OF_total_frames_v_neg = [], [], [], []        
        
        for chunk in total_chunks:
            frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                                np.ones(chunk.shape[0]).astype('int32'), 
                                (width, height, 2))   # .to_dense()
            total_frames.append(frame)

        for OF_chunk in OF_total_chunks:    

            # u
            OF_chunk_u_pos = OF_chunk[OF_chunk[:, 4] > 0]
            OF_chunk_u_neg = OF_chunk[OF_chunk[:, 4] < 0]
            OF_chunk_u_neg[:, 4] = np.abs(OF_chunk_u_neg[:, 4])            
            
            OF_frame_u_pos = sparse.COO(OF_chunk_u_pos[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk_u_pos[:,4].astype('int32'), 
                                (width, height, 2))   # .todense()
            OF_total_frames_u_pos.append(OF_frame_u_pos)
            OF_frame_u_neg = sparse.COO(OF_chunk_u_neg[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk_u_neg[:,4].astype('int32'), 
                                (width, height, 2))   # .todense()
            OF_total_frames_u_neg.append(OF_frame_u_neg)

            
            # v
            
            OF_chunk_v_pos = OF_chunk[OF_chunk[:, 5] > 0]
            OF_chunk_v_neg = OF_chunk[OF_chunk[:, 5] < 0]
            OF_chunk_v_neg[:, 5] = np.abs(OF_chunk_v_neg[:, 5]) 
            OF_frame_v_pos = sparse.COO(OF_chunk_v_pos[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk_v_pos[:,5].astype('int32'), 
                                (width, height, 2))   # .todense()
            OF_total_frames_v_pos.append(OF_frame_v_pos)   
            OF_frame_v_neg = sparse.COO(OF_chunk_v_neg[:,[0,1,3]].transpose().astype('int32'), 
                                OF_chunk_v_neg[:,5].astype('int32'), 
                                (width, height, 2))   # .todense()
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
        
        if '_4' or '-4' in events_file:     val_set = 'S4' 
        elif '_3' or '-3' in events_file:   val_set = 'S3' 
        elif '_2' or '-2' in events_file:   val_set = 'S2' 
        elif '_1' or '-1' in events_file:   val_set = 'S1'
        else: raise ValueError('Set not handled')
        
        
        
        if events_file in train_samples_4sets:

            train_file_name = filename_dst.format('4sets', 'train', val_set, row['class'])
            os.makedirs(os.path.dirname(train_file_name), exist_ok=True)
            pickle.dump(total_frames, open(train_file_name, 'wb'))
            pickle.dump(OF_total_frames_u_pos, open(train_file_name[:80]+ 'OF_u_pos_' + train_file_name[80:], 'wb'))
            pickle.dump(OF_total_frames_u_neg, open(train_file_name[:80]+ 'OF_u_neg_' + train_file_name[80:], 'wb'))
            pickle.dump(OF_total_frames_v_pos, open(train_file_name[:80]+ 'OF_v_pos_' + train_file_name[80:], 'wb'))
            pickle.dump(OF_total_frames_v_neg, open(train_file_name[:80]+ 'OF_v_neg_' + train_file_name[80:], 'wb'))

            aug_roll_total_frames = SME.nda(total_frames, 'roll')
            aug_roll_total_frames_dst = train_file_name[:80] + 'aug_roll_' + train_file_name[80:]
            if os.path.isfile(aug_roll_total_frames_dst): 
                print(aug_roll_total_frames_dst, "done.")
                pass
            else: pickle.dump(aug_roll_total_frames, open(aug_roll_total_frames_dst, 'wb'))

            aug_rotate_total_frames = SME.nda(total_frames, 'rotate')
            aug_rotate_total_frames_dst = train_file_name[:80] + 'aug_rotate_' + train_file_name[80:]
            if os.path.isfile(aug_rotate_total_frames_dst): 
                print(aug_rotate_total_frames_dst, "done.")
                pass
            else: pickle.dump(aug_rotate_total_frames, open(aug_rotate_total_frames_dst, 'wb'))

            aug_shear_total_frames = SME.nda(total_frames, 'shear')
            aug_shear_total_frames_dst = train_file_name[:80] + 'aug_shear_' + train_file_name[80:]
            if os.path.isfile(aug_shear_total_frames_dst): 
                print(aug_shear_total_frames_dst, "done.")
                pass
            else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst, 'wb'))

            #u_pos
            aug_roll_total_frames_u_pos = SME.nda(OF_total_frames_u_pos, 'roll')
            aug_roll_total_frames_dst_u_pos = train_file_name[:80] + 'aug_roll_u_pos_' + train_file_name[80:]
            if os.path.isfile(aug_roll_total_frames_dst_u_pos): 
                print(aug_roll_total_frames_dst_u_pos, "done.")
                pass
            else: pickle.dump(aug_roll_total_frames_u_pos, open(aug_roll_total_frames_dst_u_pos, 'wb'))

            aug_rotate_total_frames_u_pos = SME.nda(OF_total_frames_u_pos, 'rotate')
            aug_rotate_total_frames_dst_u_pos = train_file_name[:80] + 'aug_rotate_u_pos_' + train_file_name[80:]
            if os.path.isfile(aug_rotate_total_frames_dst_u_pos): 
                print(aug_rotate_total_frames_dst_u_pos, "done.")
                pass
            else: pickle.dump(aug_rotate_total_frames_u_pos, open(aug_rotate_total_frames_dst_u_pos, 'wb'))

            aug_shear_total_frames_u_pos = SME.nda(OF_total_frames_u_pos, 'shear')
            aug_shear_total_frames_dst_u_pos = train_file_name[:80] + 'aug_shear_u_pos_' + train_file_name[80:]
            if os.path.isfile(aug_shear_total_frames_dst_u_pos): 
                print(aug_shear_total_frames_dst_u_pos, "done.")
                pass
            else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst_u_pos, 'wb'))

            #u_neg
            aug_roll_total_frames_u_neg = SME.nda(OF_total_frames_u_neg, 'roll')
            aug_roll_total_frames_dst_u_neg = train_file_name[:80] + 'aug_roll_u_neg_' + train_file_name[80:]
            if os.path.isfile(aug_roll_total_frames_dst_u_neg): 
                print(aug_roll_total_frames_dst_u_neg, "done.")
                pass
            else: pickle.dump(aug_roll_total_frames_u_neg, open(aug_roll_total_frames_dst_u_neg, 'wb'))

            aug_rotate_total_frames_u_neg = SME.nda(OF_total_frames_u_neg, 'rotate')
            aug_rotate_total_frames_dst_u_neg = train_file_name[:80] + 'aug_rotate_u_neg_' + train_file_name[80:]
            if os.path.isfile(aug_rotate_total_frames_dst_u_neg): 
                print(aug_rotate_total_frames_dst_u_neg, "done.")
                pass
            else: pickle.dump(aug_rotate_total_frames_u_neg, open(aug_rotate_total_frames_dst_u_neg, 'wb'))

            aug_shear_total_frames_u_neg = SME.nda(OF_total_frames_u_neg, 'shear')
            aug_shear_total_frames_dst_u_neg = train_file_name[:80] + 'aug_shear_u_neg_' + train_file_name[80:]
            if os.path.isfile(aug_shear_total_frames_dst_u_neg): 
                print(aug_shear_total_frames_dst_u_neg, "done.")
                pass
            else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst_u_neg, 'wb'))


            #v_pos
            aug_roll_total_frames_v_pos = SME.nda(OF_total_frames_v_pos, 'roll')
            aug_roll_total_frames_dst_v_pos = train_file_name[:80] + 'aug_roll_v_pos_' + train_file_name[80:]
            if os.path.isfile(aug_roll_total_frames_dst_v_pos): 
                print(aug_roll_total_frames_dst_v_pos, "done.")
                pass
            else: pickle.dump(aug_roll_total_frames_v_pos, open(aug_roll_total_frames_dst_v_pos, 'wb'))

            aug_rotate_total_frames_v_pos = SME.nda(OF_total_frames_v_pos, 'rotate')
            aug_rotate_total_frames_dst_v_pos = train_file_name[:80] + 'aug_rotate_v_pos_' + train_file_name[80:]
            if os.path.isfile(aug_rotate_total_frames_dst_v_pos): 
                print(aug_rotate_total_frames_dst_v_pos, "done.")
                pass
            else: pickle.dump(aug_rotate_total_frames_v_pos, open(aug_rotate_total_frames_dst_v_pos, 'wb'))

            aug_shear_total_frames_v_pos = SME.nda(OF_total_frames_v_pos, 'shear')
            aug_shear_total_frames_dst_v_pos = train_file_name[:80] + 'aug_shear_v_pos_' + train_file_name[80:]
            if os.path.isfile(aug_shear_total_frames_dst_v_pos): 
                print(aug_shear_total_frames_dst_v_pos, "done.")
                pass
            else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst_v_pos, 'wb'))

            #v_neg
            aug_roll_total_frames_v_neg = SME.nda(OF_total_frames_v_neg, 'roll')
            aug_roll_total_frames_dst_v_neg = train_file_name[:80] + 'aug_roll_v_neg_' + train_file_name[80:]
            if os.path.isfile(aug_roll_total_frames_dst_v_neg): 
                print(aug_roll_total_frames_dst_v_neg, "done.")
                pass
            else: pickle.dump(aug_roll_total_frames_v_neg, open(aug_roll_total_frames_dst_v_neg, 'wb'))

            aug_rotate_total_frames_v_neg = SME.nda(OF_total_frames_v_neg, 'rotate')
            aug_rotate_total_frames_dst_v_neg = train_file_name[:80] + 'aug_rotate_v_neg_' + train_file_name[80:]
            if os.path.isfile(aug_rotate_total_frames_dst_v_neg): 
                print(aug_rotate_total_frames_dst_v_neg, "done.")
                pass
            else: pickle.dump(aug_rotate_total_frames_v_neg, open(aug_rotate_total_frames_dst_v_neg, 'wb'))

            aug_shear_total_frames_v_neg = SME.nda(OF_total_frames_v_neg, 'shear')
            aug_shear_total_frames_dst_v_neg = train_file_name[:80] + 'aug_shear_v_neg_' + train_file_name[80:]
            if os.path.isfile(aug_shear_total_frames_dst_v_neg): 
                print(aug_shear_total_frames_dst_v_neg, "done.")
                pass
            else: pickle.dump(aug_shear_total_frames, open(aug_shear_total_frames_dst_v_neg, 'wb'))

        if events_file in test_samples_4sets:
            test_file_name = filename_dst.format('4sets', 'test', val_set, row['class'])
            pickle.dump(total_frames, open(test_file_name, 'wb'))
