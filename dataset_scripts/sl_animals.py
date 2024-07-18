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
import SME

np.random.seed(0)

chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height, width = 128, 128
size = 0.25

# Source data folder
path_dataset = '../datasets/SL_Animals_SME_nda/'
dst_path = '../datasets/SL_Animals/'
files = os.listdir(path_dataset + 'allusers_aedat_mat/')
parser = aermanager.parsers.parse_dvs_128

# Target data folder
if not os.path.isdir(dst_path + 'SL_animal_splits'):
    os.mkdir(dst_path + 'SL_animal_splits')
    os.makedirs(dst_path + 'SL_animal_splits/' + f'dataset_4sets_{chunk_len_us}/train')
    os.makedirs(dst_path + 'SL_animal_splits/' + f'dataset_4sets_{chunk_len_us}/test')
    os.makedirs(dst_path + 'SL_animal_splits/' + f'dataset_3sets_{chunk_len_us}/train')
    os.makedirs(dst_path + 'SL_animal_splits/' + f'dataset_3sets_{chunk_len_us}/test')

test_samples_4sets = [ 'user10_indoor.aedat', 'user12_indoor.aedat', 'user14_indoor.aedat', 'user17_indoor.aedat', 'user19_sunlight.aedat', 
                      'user24_sunlight.aedat', 'user29_imse.aedat', 'user30_imse.aedat', 'user34_imse.aedat', 'user35_dc.aedat', 
                      'user36_dc.aedat', 'user37_dc.aedat', 'user38_dc.aedat', 'user42_dc.aedat', 'user57_dc.aedat' ]
train_samples_4sets = [ f for f in files if f not in test_samples_4sets ]

test_samples_3sets = [ 'user19_sunlight.aedat', 
                      'user24_sunlight.aedat', 'user29_imse.aedat', 'user30_imse.aedat', 'user34_imse.aedat', 'user35_dc.aedat', 
                      'user36_dc.aedat', 'user37_dc.aedat', 'user38_dc.aedat', 'user42_dc.aedat', 'user57_dc.aedat' ]
train_samples_3sets = [ f for f in files if f not in test_samples_3sets and '_indoor' not in f ]                      

for events_file in tqdm(files):
    if events_file == '': continue
 
    root, extension = os.path.splitext(events_file)               
    if extension == '.aedat':
        shape, events = load_events_from_file(path_dataset + 'allusers_aedat_mat/' + events_file, parser=parser)
        labels = pd.read_csv(path_dataset + 'tags_updated_19_08_2020/' + events_file.replace('.aedat', '.csv'))
        filename_dst = dst_path + 'SL_animal_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
        events_file.replace('.aedat', '_{}_{}.pckl')
    else: continue
    
    for _,row in labels.iterrows():
        
        if extension == '.aedat':
            sample_events = events[row.startTime_ev:row.endTime_ev]
            total_events = np.array([sample_events['x'], sample_events['y'], sample_events['t'], sample_events['p']]).transpose()
        else: continue

        total_chunks = []
        
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
                total_events = total_events[:max(1, chunk_inds.min())-1]
        if len(total_chunks) == 0: continue
        total_chunks = total_chunks[::-1]
            
        total_frames = []

        for chunk in total_chunks:    
            
            frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                                np.ones(chunk.shape[0]).astype('int32'), 
                                (height, width, 2))   # .todense()
            total_frames.append(frame)                       
        
        total_frames = sparse.stack(total_frames)
        total_frames = np.clip(total_frames, a_min=0, a_max=255)
        total_frames = total_frames.astype('uint8')
             

        if '_sunlight' in events_file:  val_set = 'S4' # S4 indoors with frontal sunlight
        elif '_indoor' in events_file:  val_set = 'S3' # S3 indoors neon light
        elif '_dc' in events_file:      val_set = 'S2' # S2 natural side light
        elif '_imse' in events_file:    val_set = 'S1' # S1 natural side light
        else: raise ValueError('Set not handled')
        
        if events_file in train_samples_3sets:     
            filename_dst_final = filename_dst.format('3sets', 'train', val_set, row['class'])
               
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))            
            
            # Augmentation to whole total_frames
            aug_roll_total_frames = SME.nda(total_frames, 'roll')
            pickle.dump(aug_roll_total_frames, open(filename_dst_final[:87] + 'aug_roll_' + filename_dst_final[87:], 'wb'))
            aug_rotate_total_frames = SME.nda(total_frames, 'rotate')
            pickle.dump(aug_rotate_total_frames, open(filename_dst_final[:87] + 'aug_rotate_' + filename_dst_final[87:], 'wb'))
            aug_shear_total_frames = SME.nda(total_frames, 'shear')
            pickle.dump(aug_shear_total_frames, open(filename_dst_final[:87] + 'aug_shear_' + filename_dst_final[87:], 'wb'))             
           

        if events_file in train_samples_4sets:
            filename_dst_final = filename_dst.format('4sets', 'train', val_set, row['class'])
               
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))            
            
            # Augmentation to whole total_frames
            aug_total_frames = SME.nda(total_frames, 'roll')
            pickle.dump(aug_total_frames, open(filename_dst_final[:87] + 'aug_roll_' + filename_dst_final[87:], 'wb'))
            aug_total_frames = SME.nda(total_frames, 'rotate')
            pickle.dump(aug_total_frames, open(filename_dst_final[:87] + 'aug_rotate_' + filename_dst_final[87:], 'wb'))
            aug_total_frames = SME.nda(total_frames, 'shear')
            pickle.dump(aug_total_frames, open(filename_dst_final[:87] + 'aug_shear_' + filename_dst_final[87:], 'wb'))


        if events_file in test_samples_3sets:
            filename_dst_final = filename_dst.format('3sets', 'test', val_set, row['class'])              
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))
                                    
        if events_file in test_samples_4sets:       
            filename_dst_final = filename_dst.format('4sets', 'test', val_set, row['class'])              
            pickle.dump(total_frames, open(filename_dst_final, 'wb'))        
