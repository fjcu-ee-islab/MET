import numpy as np
import os
import h5py
import csv    

from tqdm import tqdm
from scipy.io import savemat

mapping = { 0 :'comeHere',
            1 :'left_swipe',
            2 :'right_swipe',
            3 :'rotation_CCW',
            4 :'rotation_CW',
            5 :'swipeDown',
            6 :'swipeUP',
            7 :'swipeV',
            8 :'X',
            9 :'Z'}

if __name__ == '__main__':
    PATH = os.path.dirname(os.path.realpath(__file__))
    
    ### IITM
    OF_input_class_dir = '/home/eden/Desktop/MEVT datasets/IITM_OF/OF_input/'
    classes = os.listdir(OF_input_class_dir)
    
    for c in classes:
        
        if c == 'comeHere':           c_number = '0'
        elif c == 'left_swipe':       c_number = '1'
        elif c == 'right_swipe':      c_number = '2'
        elif c == 'rotation_CCW':     c_number = '3'
        elif c == 'rotation_CW':      c_number = '4'
        elif c == 'swipeDown':        c_number = '5'
        elif c == 'swipeUP':          c_number = '6'
        elif c == 'swipeV':           c_number = '7'
        elif c == 'X':                c_number = '8'
        elif c == 'Z':                c_number = '9'

        
        OF_input_dir = OF_input_class_dir + '/' + c
        OF_input_aedat_file_dir = '/home/eden/Desktop/MEVT datasets/IITM_OF/AEDAT4.0/' + c + '/'     
        OF_visual_dir = '/home/eden/Desktop/MEVT datasets/IITM_OF/OF_saved/' + c_number + '/visualizations/'
        OF_file_output_dir = '/home/eden/Desktop/MEVT datasets/IITM_OF/IITM_total_flow/'+ c_number + '/'
        #breakpoint()
        '''
        ### SL_Animals_DVS
        OF_input_dir = PATH + '/data/SL_Animals/'
        OF_input_aedat_file_dir = PATH + '/data/SL_Animals/aedat/'     
        OF_visual_dir = PATH + '/saved/SL_Animal_DVS_OF_output/visualizations'
        OF_file_output_dir = PATH + '/saved/SL_Animals_total_flow'
        '''
        
        '''
        ### DVS_Gesture
        OF_input_dir = PATH + '/data/DVS_Gesture/'
        OF_input_aedat_file_dir = '/home/eden/Desktop/nvme1n1/MotionEventTransformer/datasets/DvsGesture_SME_nda/'     
        OF_visual_dir = PATH + '/saved/DVS_Gesture_OF_Output/visualizations/'
        OF_file_output_dir = PATH + '/saved/DVS_Gesture_total_flow/'
        '''
            
        files = os.listdir(OF_input_aedat_file_dir)
        if not os.path.isdir(OF_file_output_dir):
            os.mkdir(OF_file_output_dir)
        
        aedat_files = [] 
        for i in range(len(files)):
            if '.aedat' in files[i]:
                aedat_files.append(files[i])
        
        OF_events_struct = [("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("u", np.float32), ("v", np.float32) ]
        
        for events_file in tqdm(aedat_files):
            if events_file == '': continue
            root, extension = os.path.splitext(events_file) 
            if c_number == '6' and root == '00044': continue
            if c_number == '6' and root == '00101': continue
            if c_number == '6' and root == '00033': continue
            if c_number == '6' and root == '00006': continue
            print("\nConvert total flow: " + root + '.h5')
            
            OF_File_input_events = OF_input_dir + '/test/' + root + '/events_left/events.h5'
            with h5py.File(OF_File_input_events, "r") as f: 
                OF_File_ts_offset = f['t_offset'][()]
                #print("Keys: %s" % f.keys())
            
            OF_File_flow_timestamps = OF_input_dir + '/test/' + root + '/test_forward_flow_timestamps.csv'
            flow_timestamps_array = np.loadtxt(OF_File_flow_timestamps, skiprows=1, delimiter=',')
            from_timestamp_us_array = flow_timestamps_array[:,0]
            to_timestamp_us_array = flow_timestamps_array[:,1]
            file_index_array = flow_timestamps_array[:,2]
        
            flow_dir = OF_visual_dir + '/' + root + '/'
            flow_files = os.listdir(flow_dir)
            
            OF_file_output = OF_file_output_dir + '/' + 'total_flow_' + root + '.h5'
            with h5py.File(OF_file_output, 'w') as hf:
                for flow_file in tqdm(flow_files):
                    root_flow, extension_flow = os.path.splitext(flow_file) 
                    if extension_flow == '.h5':
                        with h5py.File(flow_dir + flow_file, "r") as ff: 
                            u = ff['flow']['u'][()]
                            v = ff['flow']['v'][()]
                            #breakpoint()
                            #print("")
                        print("Target file: ", OF_file_output)
                        print("Processing... ")  
                        flow_index = int(flow_file.split('_')[1])                
                        ts_floor_from_index = from_timestamp_us_array[flow_index//10]

                        ts_ceil_from_index = to_timestamp_us_array[flow_index//10]
                        ts = ts_floor_from_index + (ts_ceil_from_index - ts_floor_from_index) * ((flow_index - flow_index//10*10) / (((flow_index//10+1)*10) - (flow_index//10*10)))

                        # 創建坐標網格
                        x, y = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing='ij')

                        # 扁平化數組和坐標
                        x_flat = x.ravel()
                        y_flat = y.ravel()
                        u_flat = u.ravel()
                        v_flat = v.ravel()

                        # 定義結構化數據型別
                        OF_events_struct = np.dtype([("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("u", np.float32), ("v", np.float32)])

                        # 創建一個空的結構化數組並填充它
                        num_events = u_flat.shape[0]
                        OF_events = np.zeros(num_events, dtype=OF_events_struct)
                        OF_events['x'] = x_flat
                        OF_events['y'] = y_flat
                        OF_events['t'] = np.full(num_events, ts, dtype=np.uint32)
                        OF_events['u'] = u_flat
                        OF_events['v'] = v_flat

                        # 儲存到 HDF5 檔案
                        #print(ts, OF_File_ts_offset, ts-OF_File_ts_offset)
                        hf.create_dataset(str(int(ts)), data=OF_events)  
            
                    else: continue

            print("Complete! Save path: ", OF_file_output, '\n')
