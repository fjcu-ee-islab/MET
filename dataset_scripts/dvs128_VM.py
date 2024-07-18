import numpy as np
from tqdm import tqdm
import pickle
import os
import sparse
import SME

chunk_len_ms = 96
chunk_len_us = chunk_len_ms*1000
height = width = 128



def dvs128_SME(event_files, path_dataset_src, path_dataset_dst, mode):
    total_chunk_time_window_length = 0
    total_chunk_time_window_length_total = 0
    total_chunk_num = 0
    for ef in tqdm(event_files):
        
        total_events, label = pickle.load(open(path_dataset_src+ef, 'rb'))
        total_events = total_events.astype('int32')
        
        total_chunks = []
        while total_events.shape[0] > 0:
            end_t = total_events[-1][2]
            chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
            
            if total_events.shape[1] == 4:
                if len(chunk_inds) <= 4: 
                    pass
                else:
                    total_chunks.append(total_events[chunk_inds])

                    time_window_length = total_events[chunk_inds[-1]][2] - total_events[chunk_inds[0]][2]
                    total_chunk_time_window_length_total = total_chunk_time_window_length_total + time_window_length
                    total_chunk_num += 1

            elif total_events.shape[1] == 6:
                if len(chunk_inds) <= 0: 
                    pass
                else:
                    total_chunks.append(total_events[chunk_inds])

                    time_window_length = total_events[chunk_inds[-1]][2] - total_events[chunk_inds[0]][2]
                    total_chunk_time_window_length_total = total_chunk_time_window_length_total + time_window_length
                    total_chunk_num += 1
                  
            
            total_events = total_events[:max(1, chunk_inds.min())-1]
        
        
        if len(total_chunks) == 0: continue
        total_chunks = total_chunks[::-1]
            
        total_frames = []
        total_speed = []

        for chunk in total_chunks:

            #[1,0,3]
            if(chunk.shape[1] == 4):
                frame = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'),
                                np.ones(chunk.shape[0]).astype('int32'),
                                (height, width, 2))   # .to_dense()
                total_frames.append(frame)
            if(chunk.shape[1] == 6):
                frame = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'),
                                np.ones(chunk.shape[0]).astype('int32'),
                                (height, width, 2))   # .to_dense()
                total_frames.append(frame)                
                frame_speed = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'), 
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

        pickle.dump(total_frames, open(path_dataset_dst + ef, 'wb'))
       
 
        # SME
        if mode == 'train':
            aug_total_frames = SME.nda(total_frames, 'roll')
            pickle.dump(aug_total_frames, open(path_dataset_dst + 'aug_roll_' +ef, 'wb'))  

            aug_total_frames = SME.nda(total_frames, 'rotate')
            pickle.dump(aug_total_frames, open(path_dataset_dst + 'aug_rotate_' +ef, 'wb'))  

            aug_total_frames = SME.nda(total_frames, 'shear')
            pickle.dump(aug_total_frames, open(path_dataset_dst + 'aug_shear_' +ef, 'wb'))
            
            if(isinstance(total_speed, sparse.COO)):
                speed_filename = (path_dataset_dst + "speed_" + ef )
                pickle.dump(total_speed, open(speed_filename, 'wb'))
                
                # Augmentation to whole total_speed
                aug_roll_total_speed = SME.nda(total_speed, 'roll')
                speed_filename = (path_dataset_dst + "aug_roll_speed_" + ef )
                pickle.dump(aug_roll_total_speed, open(speed_filename, 'wb'))
                
                aug_rotate_total_speed = SME.nda(total_speed, 'rotate')
                speed_filename = (path_dataset_dst + "aug_rotate_speed_" + ef )
                pickle.dump(aug_rotate_total_speed, open(speed_filename, 'wb'))
                
                aug_shear_total_speed = SME.nda(total_speed, 'shear')
                speed_filename = (path_dataset_dst + "aug_shear_speed_" + ef )
                pickle.dump(aug_shear_total_speed, open(speed_filename, 'wb'))

    print("total_chunk_num = ", total_chunk_num)
    print("total_chunk_time_window_length_total = ", total_chunk_time_window_length_total)
    print("Average time window length = ", total_chunk_time_window_length_total / total_chunk_num/1000, "ms")
    breakpoint()

            


for mode in ['test']:
    # Read dataset filenames
    if mode == 'train':
        # Source data folder
        path_dataset_src = '../datasets/DvsGesture_SME_nda/clean_dataset/train/'
        # Target data folder
        path_dataset_dst = '../datasets/DvsGesture_SME_nda/clean_dataset_frames_{}/train/'.format(chunk_len_us)
    elif mode == 'test':
        path_dataset_src = '../datasets/DvsGesture_SME_nda/clean_dataset/test/'
        path_dataset_dst = '../datasets/DvsGesture_SME_nda/clean_dataset_frames_{}/test/'.format(chunk_len_us)

    event_files = os.listdir(path_dataset_src)
    if not os.path.isdir(path_dataset_dst): os.makedirs(path_dataset_dst)

    dvs128_SME(event_files, path_dataset_src, path_dataset_dst, mode)
