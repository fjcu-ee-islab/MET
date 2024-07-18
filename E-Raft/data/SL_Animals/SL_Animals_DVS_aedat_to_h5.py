
import os
import numpy as np
import aermanager
from aermanager.aerparser import load_events_from_file
from tqdm import tqdm

import h5py
import csv

def create_ms_to_idx_mapping(t, ms_list):
    # Initialize an empty array to hold the mapping, filled with -1 as placeholders
    ms_to_idx = np.full(len(ms_list), -1, dtype=np.int)
    
    # Initialize pointers
    idx = 0
    t_len = len(t)
    
    for i, ms in enumerate(ms_list):
        target_time = ms * 1000  # Convert ms to μs
        
        # Find the index that satisfies condition (1)
        while idx < t_len and t[idx] < target_time:
            idx += 1
        
        # If no such index exists, break the loop
        if idx >= t_len:
            break
        
        # Store the mapping
        ms_to_idx[i] = idx
    
    return ms_to_idx

def find_closest_ts_every_n(path_image_timestamps_file, t):
    
    n = 50000
    img_ts_file = open(path_image_timestamps_file, 'w+')
    img_ts = []
    max_t = t.max()
    min_t = t.min()
    # Initialize pointers
    t_idx = 0
    target_idx = 0
    target_time = min_t
    end_time = max_t
    t_len = t.shape[0]
    
    while(target_time <= end_time):
        while t_idx < t_len and t[t_idx] <= target_time:
            t_idx += 1
            
        if t_idx >= t_len:
            break
        
        if img_ts == []:
            img_ts.append((t[t_idx]+100000).astype('int').astype('str'))
            img_ts_file.write((t[t_idx]+100000).astype('int').astype('str')+'\n')
        else:
            if (t[t_idx]+100000).astype('int').astype('str') != img_ts[-1]:     
                img_ts.append((t[t_idx]+100000).astype('int').astype('str'))
                img_ts_file.write((t[t_idx]+100000).astype('int').astype('str')+'\n')
            else: pass

        target_idx += 1
        target_time = min_t + target_idx * n
        
        #if(target_time > end_time): breakpoint()
    return img_ts

def write_timestamp_csv_v1(path_csv_file, img_ts):
    with open(path_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['from_timestamp_us', 'to_timestamp_us', 'file_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        idx = 0
        num_idx = 0
        to_idx = 0
        
        while(idx < len(img_ts)):
            from_idx = idx
            if (idx+1 < len(img_ts)): to_idx = idx+2
            elif (idx+1 == len(img_ts)):  to_idx = idx
            #if idx == 1900: breakpoint()
            if (from_idx >= to_idx): breakpoint()
            
            time_from_timestamp_us = img_ts[from_idx]            
            time_to_timestamp_us = img_ts[to_idx]
            
            writer.writerow({'from_timestamp_us': time_from_timestamp_us, 'to_timestamp_us': time_to_timestamp_us, 'file_index': num_idx*10})
            idx += 10
            num_idx += 1
    
    return 0


def write_timestamp_csv(path_csv_file, img_ts):
    with open(path_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['from_timestamp_us', 'to_timestamp_us', 'file_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        idx = 0
        num_idx = 0
        
        while(idx < len(img_ts)-2):
            time_from_timestamp_us = img_ts[idx]            
            time_to_timestamp_us = img_ts[idx+2]
            
            '''
            while(time_from_timestamp_us == time_to_timestamp_us):
                idx += 1
                if(idx < len(img_ts)-2):
                    time_to_timestamp_us = img_ts[idx+2]
                else: break
            if(idx >= len(img_ts)-2): break                    
            '''
            writer.writerow({'from_timestamp_us': time_from_timestamp_us, 'to_timestamp_us': time_to_timestamp_us, 'file_index': num_idx*10})
            idx += 10
            num_idx += 1
    
    return 0

PATH = os.path.dirname(os.path.realpath(__file__))
path_aedat = PATH + '/aedat/'
files = os.listdir(path_aedat)
parser = aermanager.parsers.parse_dvs_128

'''
example_h5_file = '/root/notebooks/eden/E-RAFT/data/dsec_test/test/zurich_city_12_a/events_left/events.h5'

with h5py.File(example_h5_file, "r") as f: 
    print("Keys: %s" % f.keys())
'''

for events_file in tqdm(files):
    shape, all_events = load_events_from_file(path_aedat + events_file, parser=parser)
    file_name = events_file.split(".")[0]
    path_h5_dir = PATH + '/test/' + file_name
    if not os.path.isdir(path_h5_dir):
        os.mkdir(path_h5_dir)

    path_h5_events_dir = path_h5_dir + '/events_left/'
    if not os.path.isdir(path_h5_events_dir):
        os.mkdir(path_h5_events_dir)

    path_h5_file =  path_h5_events_dir + 'events.h5'
    f = h5py.File(path_h5_file, 'w')

    t_offset_value = all_events['t'].min()-5
    t_offset = f.create_dataset('t_offset', data = t_offset_value)
    new_t = all_events['t'] - t_offset

    events = f.create_group('events')
    x = events.create_dataset('x', data=all_events['x'])
    y = events.create_dataset('y', data=all_events['y'])
    t = events.create_dataset('t', data=new_t)
    p = events.create_dataset('p', data=all_events['p'])

    ms_max = int(t[:].max()/1000)+1
    ms = np.arange(ms_max)
    ms_to_idx_value = create_ms_to_idx_mapping(new_t, ms)    

    ms_to_idx = f.create_dataset('ms_to_idx', data = ms_to_idx_value) 
    print('H5 file completed: ', path_h5_file)
    
    path_image_timestamps_file = path_h5_dir + '/image_timestamps.txt'

    img_ts = find_closest_ts_every_n(path_image_timestamps_file, all_events['t'])
    print('Image_timestamps_file completed: ', path_image_timestamps_file)

    path_csv_file = path_h5_dir + '/test_forward_flow_timestamps.csv'
    write_timestamp_csv(path_csv_file, img_ts)
    print('Test_forward_flow_timestamps.csv completed: ', path_csv_file)
            
