import numpy as np
from tqdm import tqdm
import pickle
import os
import sparse
import SME

chunk_len_ms = 24
chunk_len_us = chunk_len_ms*1000
height = width = 128

def dvs128_SME(event_files, path_dataset_src, path_dataset_dst, mode):
    total_chunk_time_window_length = 0
    total_chunk_time_window_length_total = 0
    total_chunk_num = 0
    for ef in tqdm(event_files):
        if '_OF' not in ef: 
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
                        
                total_events = total_events[:max(1, chunk_inds.min())-1]

            if len(total_chunks) == 0: continue
            total_chunks = total_chunks[::-1]
                
            total_frames = []

            for chunk in total_chunks:
                frame = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'),
                                np.ones(chunk.shape[0]).astype('int32'),
                                (height, width, 2))   # .to_dense()
                total_frames.append(frame)
            
            total_frames = sparse.stack(total_frames)
            total_frames = np.clip(total_frames, a_min=0, a_max=255)
            total_frames = total_frames.astype('uint8')

            pickle.dump(total_frames, open(path_dataset_dst + ef, 'wb'))
            
            # NDA_Augmentation
            if mode == 'train': 
                aug_total_frames = SME.nda(total_frames, 'roll')
                pickle.dump(aug_total_frames, open(path_dataset_dst + 'aug_roll_' +ef, 'wb'))  

                aug_total_frames = SME.nda(total_frames, 'rotate')
                pickle.dump(aug_total_frames, open(path_dataset_dst + 'aug_rotate_' +ef, 'wb'))  

                aug_total_frames = SME.nda(total_frames, 'shear')
                pickle.dump(aug_total_frames, open(path_dataset_dst + 'aug_shear_' +ef, 'wb'))
        
        elif '_OF' in ef:
            if mode != 'train': continue
            OF_events, label = pickle.load(open(path_dataset_src+ef, 'rb'))
            #OF_events = OF_events.astype('int32')
            
            OF_total_chunks = []
            
            while OF_events.shape[0] > 0:
                end_t = OF_events[-1][2]
                OF_chunk_inds = np.where(OF_events[:,2] >= end_t - chunk_len_us)[0]
                
                if OF_events.shape[1] == 6:
                    if len(OF_chunk_inds) <= 4: 
                        pass
                    else:
                        OF_total_chunks.append(OF_events[OF_chunk_inds])

                        time_window_length = OF_events[OF_chunk_inds[-1]][2] - OF_events[OF_chunk_inds[0]][2]
                        total_chunk_time_window_length_total = total_chunk_time_window_length_total + time_window_length
                        total_chunk_num += 1
                    
                OF_events = OF_events[:max(1, OF_chunk_inds.min())-1]

            if len(OF_total_chunks) == 0: 
                breakpoint()
                continue
            OF_total_chunks = OF_total_chunks[::-1]
                
            OF_total_frames_u_pos, OF_total_frames_u_neg, OF_total_frames_v_pos, OF_total_frames_v_neg = [], [], [], []


            for OF_chunk in OF_total_chunks:    

                # u            
                # 使用布林索引來找到正值和負值
                OF_chunk_u_pos = OF_chunk[OF_chunk[:, 4] >= 0]
                OF_chunk_u_neg = OF_chunk[OF_chunk[:, 4] <= 0]
                OF_chunk_u_neg[:, 4] = np.abs(OF_chunk_u_neg[:, 4])            
                
                OF_frame_u_pos = sparse.COO(OF_chunk_u_pos[:,[0,1,3]].transpose().astype('int32'), 
                                    OF_chunk_u_pos[:,4].astype('int32'), 
                                    (height, width, 2))   # .todense()
                OF_total_frames_u_pos.append(OF_frame_u_pos)
                OF_frame_u_neg = sparse.COO(OF_chunk_u_neg[:,[0,1,3]].transpose().astype('int32'), 
                OF_chunk_u_neg[:,4].astype('int32'), 
                                    (height, width, 2))   # .todense()
                OF_total_frames_u_neg.append(OF_frame_u_neg)

                # v
                # 使用布林索引來找到正值和負值
                OF_chunk_v_pos = OF_chunk[OF_chunk[:, 5] >= 0]
                OF_chunk_v_neg = OF_chunk[OF_chunk[:, 5] <= 0]
                OF_chunk_v_neg[:, 5] = np.abs(OF_chunk_v_neg[:, 5]) 
                OF_frame_v_pos = sparse.COO(OF_chunk_v_pos[:,[0,1,3]].transpose().astype('int32'), 
                                    OF_chunk_v_pos[:,5].astype('int32'), 
                                    (height, width, 2))   # .todense()
                OF_total_frames_v_pos.append(OF_frame_v_pos)   
                OF_frame_v_neg = sparse.COO(OF_chunk_v_neg[:,[0,1,3]].transpose().astype('int32'), 
                                    OF_chunk_v_neg[:,5].astype('int32'), 
                                    (height, width, 2))   # .todense()
                OF_total_frames_v_neg.append(OF_frame_v_neg)   
                
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
            
            pickle.dump(OF_total_frames_u_pos, open(path_dataset_dst + 'OF_u_pos_' + ef, 'wb'))
            pickle.dump(OF_total_frames_u_neg, open(path_dataset_dst + 'OF_u_neg_' + ef, 'wb'))
            pickle.dump(OF_total_frames_v_pos, open(path_dataset_dst + 'OF_v_pos_' + ef, 'wb')) 
            pickle.dump(OF_total_frames_v_neg, open(path_dataset_dst + 'OF_v_neg_' + ef, 'wb'))  

                
            # NDA_Augmentation
            if mode == 'train': 
                # Augmentation to whole OF_total_frames_u_pos
                aug_roll_OF_total_frames_u_pos = SME.nda(OF_total_frames_u_pos, 'roll')
                pickle.dump(aug_roll_OF_total_frames_u_pos, open(path_dataset_dst + 'aug_roll_OF_u_pos_' + ef, 'wb'))
                aug_rotate_OF_total_frames_u_pos = SME.nda(OF_total_frames_u_pos, 'rotate')
                pickle.dump(aug_rotate_OF_total_frames_u_pos, open(path_dataset_dst + 'aug_rotate_OF_u_pos_' + ef, 'wb'))
                aug_shear_OF_total_frames_u_pos = SME.nda(OF_total_frames_u_pos, 'shear')
                pickle.dump(aug_shear_OF_total_frames_u_pos, open(path_dataset_dst + 'aug_shear_OF_u_pos_' + ef, 'wb')) 
                
                # Augmentation to whole OF_total_frames_u_neg
                aug_roll_OF_total_frames_u_neg = SME.nda(OF_total_frames_u_neg, 'roll')
                pickle.dump(aug_roll_OF_total_frames_u_neg, open(path_dataset_dst + 'aug_roll_OF_u_neg_' + ef, 'wb'))
                aug_rotate_OF_total_frames_u_neg = SME.nda(OF_total_frames_u_neg, 'rotate')
                pickle.dump(aug_rotate_OF_total_frames_u_neg, open(path_dataset_dst + 'aug_rotate_OF_u_neg_' + ef, 'wb'))
                aug_shear_OF_total_frames_u_neg = SME.nda(OF_total_frames_u_neg, 'shear')
                pickle.dump(aug_shear_OF_total_frames_u_neg, open(path_dataset_dst + 'aug_shear_OF_u_neg_' + ef, 'wb'))             

                # Augmentation to whole OF_total_frames_v_pos
                aug_roll_OF_total_frames_v_pos = SME.nda(OF_total_frames_v_pos, 'roll')
                pickle.dump(aug_roll_OF_total_frames_v_pos, open(path_dataset_dst + 'aug_roll_OF_v_pos_' + ef, 'wb'))
                aug_rotate_OF_total_frames_v_pos = SME.nda(OF_total_frames_v_pos, 'rotate')
                pickle.dump(aug_rotate_OF_total_frames_v_pos, open(path_dataset_dst + 'aug_rotate_OF_v_pos_' + ef, 'wb'))
                aug_shear_OF_total_frames_v_pos = SME.nda(OF_total_frames_v_pos, 'shear')
                pickle.dump(aug_shear_OF_total_frames_v_pos, open(path_dataset_dst + 'aug_shear_OF_v_pos_' + ef, 'wb'))              

                # Augmentation to whole OF_total_frames_v_neg
                aug_roll_OF_total_frames_v_neg = SME.nda(OF_total_frames_v_neg, 'roll')
                pickle.dump(aug_roll_OF_total_frames_v_neg, open(path_dataset_dst + 'aug_roll_OF_v_neg_' + ef, 'wb'))
                aug_rotate_OF_total_frames_v_neg = SME.nda(OF_total_frames_v_neg, 'rotate')
                pickle.dump(aug_rotate_OF_total_frames_v_neg, open(path_dataset_dst + 'aug_rotate_OF_v_neg_' + ef, 'wb'))
                aug_shear_OF_total_frames_v_neg = SME.nda(OF_total_frames_v_neg, 'shear')
                pickle.dump(aug_shear_OF_total_frames_v_neg, open(path_dataset_dst + 'aug_shear_OF_v_neg_' + ef, 'wb'))  

        else: raise('Not implemented error.')

    print("total_chunk_num = ", total_chunk_num)
    print("total_chunk_time_window_length_total = ", total_chunk_time_window_length_total)
    print("Average time window length = ", total_chunk_time_window_length_total / total_chunk_num/1000, "ms")



for mode in ['train']:
    # Read dataset filenames
    if mode == 'train':
        # Source data folder
        path_dataset_src = '../datasets/DvsGesture_OF_nda/clean_dataset/train/'
        # Target data folder
        path_dataset_dst = '../datasets/DvsGesture_OF_nda/clean_dataset_frames_{}/train/'.format(chunk_len_us)
    elif mode == 'test':
        path_dataset_src = '../datasets/DvsGesture_OF_nda/clean_dataset/test/'
        path_dataset_dst = '../datasets/DvsGesture_OF_nda/clean_dataset_frames_{}/test/'.format(chunk_len_us)

    event_files = os.listdir(path_dataset_src)
    if not os.path.isdir(path_dataset_dst): os.makedirs(path_dataset_dst)

    dvs128_SME(event_files, path_dataset_src, path_dataset_dst, mode)
