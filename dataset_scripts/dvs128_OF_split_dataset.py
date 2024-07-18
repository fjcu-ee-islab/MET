import aermanager
from aermanager.aerparser import load_events_from_file
import struct
import numpy as np
import scipy.io
import pandas as pd
from tqdm import tqdm
import pickle
import os
import h5py
import SME

# Load and stor dataset event samples
def store_samples(events_files, mode):
    for events_file in tqdm(events_files):
        if events_file == '': continue
        root, extension = os.path.splitext(events_file)
     
        if extension == '.aedat':
            labels = pd.read_csv(events_file_dir + events_file.replace('.aedat', '_labels.csv'))
            shape, events = load_events_from_file(events_file_dir + events_file, parser=aermanager.parsers.parse_dvs_ibm)

            OF_file = OF_dir +'total_flow_' +root +'.h5'
            #print("Opening ", OF_file)

            with h5py.File(OF_file, 'r') as hf:
                ts_array = np.array(list(hf.keys())).astype('int')
                sorted_ts_array = np.sort(ts_array)
                
                startTime = sorted_ts_array[0]
                endTime = sorted_ts_array[-1]
                ts_array_index_in_range = np.where((sorted_ts_array>=startTime) & (sorted_ts_array<=endTime))
                ts_array_in_range = sorted_ts_array[ts_array_index_in_range]
                
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
            #breakpoint()

        # Load user samples
        # Class 0 for non action
        time_segment_class = []     # [(t_init, t_end, class)]
        time_segment_class_OF = []     # [(t_init, t_end, class)]
        prev_event = 0
        for _,row in labels.iterrows():
            if (row.startTime_usec-1 - prev_event) > 0: time_segment_class.append((prev_event, row.startTime_usec-1, 0, (row.startTime_usec-1 - prev_event)/1000))
            if ((row.endTime_usec - row.startTime_usec)) > 0: time_segment_class.append((row.startTime_usec, row.endTime_usec, row['class'], (row.endTime_usec - row.startTime_usec)/1000))
            prev_event = row.endTime_usec + 1

        time_segment_class.append((prev_event, np.inf, 0))
        time_segment_class_OF.append((prev_event, np.inf, 0))

        total_events = []
        total_OF_events = []
        curr_event = []
        curr_OF_event = []

        # for e in tqdm(events):
        for i, e in enumerate(events):
            if e[2] >= time_segment_class[0][1]:
                # Store event
                if len(curr_event) > 1:total_events.append((time_segment_class[0], len(curr_event), np.array(curr_event)))
                else:
                    if extension == '.aedat':
                        print(' ** EVENT ERROR:', events_file.replace('.aedat','') + '_num{:02d}_label{:02d}.pckl'.format(i, time_segment_class[0][2]), time_segment_class[0], len(curr_event))
                time_segment_class = time_segment_class[1:]
                curr_event = []
            curr_event.append(list(e))

        for i, e in enumerate(OF_events):
            if e[2] >= time_segment_class_OF[0][1]:
                # Store event
                if len(curr_OF_event) > 1:total_OF_events.append((time_segment_class_OF[0], len(curr_OF_event), np.array(curr_OF_event)))
                #else:
                #    if extension == '.aedat':
                #        print(' ** EVENT ERROR:', events_file.replace('.aedat','') + '_num{:02d}_label{:02d}.pckl'.format(i, time_segment_class[0][2]), time_segment_class[0], len(curr_event))
                time_segment_class_OF = time_segment_class_OF[1:]
                curr_OF_event = []
            curr_OF_event.append(list(e))

        if len(curr_event) > 1: total_events.append((time_segment_class[0], len(curr_event), np.array(curr_event)))
        if len(curr_OF_event) > 1: total_OF_events.append((time_segment_class_OF[0], len(curr_OF_event), np.array(curr_OF_event)))
        time_segment_class = time_segment_class[1:]
        time_segment_class_OF = time_segment_class_OF[1:]
        curr_event = []
        curr_OF_event = []
        for i, (meta, _, events) in enumerate(total_events):
            if extension == '.aedat':
                pickle.dump((events, meta[2]), 
                        open(os.path.join(path_dataset_dst, mode, events_file.replace('.aedat','') + '_num{:02d}_label{:02d}.pckl'.format(i, meta[2])), 'wb'))
        for i_OF, (meta_OF, _, OF_events) in enumerate(total_OF_events):
            if extension == '.aedat':
                pickle.dump((OF_events, meta_OF[2]), 
                        open(os.path.join(path_dataset_dst, mode, events_file.replace('.aedat','') + '_OF_num{:02d}_label{:02d}.pckl'.format(i_OF, meta_OF[2])), 'wb'))
        
if __name__ == '__main__':
    # Source data folder
    #events_file_dir = '../datasets/DvsGesture_SME_nda/'
    events_file_dir = '../datasets/DvsGesture/'
    OF_dir = '../datasets/DVS_Gesture_total_flow_10ms/'
    path_dataset = '../datasets/DvsGesture_OF/'

    # Target data folder
    path_dataset_dst = '../datasets/DvsGesture_OF/clean_dataset/'

    train_files, test_files = 'trials_to_train.txt', 'trials_to_test.txt'
    with open(events_file_dir + train_files, 'r') as f: train_files = f.read().splitlines()
    with open(events_file_dir + test_files, 'r') as f: test_files = f.read().splitlines()

    store_samples(train_files, 'train')
    store_samples(test_files, 'test')

