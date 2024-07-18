import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from dv import AedatFile

import aermanager
from aermanager.aerparser import load_events_from_file
from aermanager.parsers import parse_header_from_file, get_aer_events_from_file

from tqdm import tqdm
from scipy.io import savemat, loadmat

import cv2
import pickle
import sparse
import shutil

def my_parse_aedat4(in_file):
    """
    Get the aer events from version 4 of .aedat file

    Args:
        in_file: str The name of the .aedat file
    Returns:
        shape (Tuple): Shape of the sensor (height, width)
        xytp:   numpy structured array of events
    """
    with AedatFile(in_file) as f:
        shape = f['events'].size
        x, y, t, p = [], [], [], []
        for packet in f.numpy_packet_iterator("events"):
            x.append(packet["x"])
            y.append(packet["y"])
            t.append(packet["timestamp"])
            p.append(packet["polarity"])
    x = np.hstack(x)
    y = np.hstack(y)
    t = np.hstack(t)
    p = np.hstack(p)

    print("t = ", t)
    print("x = ", x)
    print("y = ", y)
    print("p = ", p)
    return x, y, t, p

def my_parse_dvs_128(filename):
    """
    Get the aer events from DVS with resolution of rows and cols are (128, 128)

    Args:
        filename: filename
    Returns:
        shape (tuple):
            (height, width) of the sensor array
        xytp: numpy structured array of events
    """
    data_version, data_start = parse_header_from_file(filename)
    all_events = get_aer_events_from_file(filename, data_version, data_start)
    all_addr = all_events["address"]
    t = all_events["timeStamp"]

    x = (all_addr >> 8) & 0x007F
    y = (all_addr >> 1) & 0x007F
    p = all_addr & 0x1

    print("all_addr = ", all_addr)
    print("x = ", x)
    print("y = ", y)
    print("p = ", p)
    
    shape = (128, 128)
    return shape, x, y, t, p

def ReadAedat(test_fn, path_dataset, result_path, dataset_name):
    files = os.listdir(path_dataset)
    
    x, y, t, p = [], [], [], []

    for events_file in tqdm(files):
        
        if 'aedat' not in events_file: continue
        #test_fn = 'user30_imse_S1_1' # default: 'user30_imse_S1_1'

        if dataset_name == "SL_Animals_dvs":
            # "SL_Animals_dvs" datasets data uses "my_parse_dvs_128" function
            shape, x, y, t, p = my_parse_dvs_128(path_dataset + events_file)
        elif dataset_name == "Fall Detection" or "UCF11" or "IITM":
            # "Fall Detection" datasets data uses "my_parse_aedat4" function
            x, y, t, p = my_parse_aedat4(path_dataset + events_file)
        else:
            assert("Non implement error")
    
        events = {"x": x, "y": y, "t": t, "p": p}
        print(path_dataset + events_file)
        print(x.size)
        print(y.size)
        print(t.size)
        print(p.size)

        return x, y, t, p

def activate_by_patch_mask(frame, frame_rgb, i, j, activated_patch_mask_path, heap_map_path):

    patch_num = 0

    # Define patch mask 
    patch_file = activated_patch_mask_path + '/patch_mask_visual/patch_'+str(patch_num)+'_ts_'+str(i)+'.mat'
    patch = loadmat(patch_file)
    patch_centers = patch['xy']

    patch_size = 6

    # Initialize a blank mask of the same size as the frame
    patch_mask = np.zeros_like(frame) # for grayscale event frame
    patch_mask_window = np.zeros_like(frame)
    patch_mask_window_activated = np.zeros_like(frame)
    Activated_patch_frame = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 3), dtype=np.uint8) # for rgb heatmap

    # Apply the patch masks
    for pixel_counter, center in enumerate(patch_centers):
        x, y = center+5
        # Calculate the top left corner of the mask
        x_start = max(x - patch_size//2, 0)
        y_start = max(y - patch_size//2, 0)

        # Calculate the bottom right corner of the mask
        x_end = min(x + patch_size//2, frame_rgb.shape[0])
        y_end = min(y + patch_size//2, frame_rgb.shape[1])

        heatmap_rgb_img = cv2.imread(heap_map_path+'patch_'+str(patch_num)+'_test_cam_'+str(i)+'.jpg')
        
        pixelcolor_heatmap = heatmap_rgb_img[0][pixel_counter]
        
        patch_mask_window[x_start:x_end, y_start] = 1
        patch_mask_window[x_start:x_end, y_end] = 1
        patch_mask_window[x_start, y_start:y_end] = 1
        patch_mask_window[x_end, y_start:y_end] = 1

        # Apply the mask to the Activated_patch_frame        
        if frame[x_start:x_end, y_start:y_end].any() == 0: pass
        else: 
            patch_mask[x_start:x_end, y_start:y_end] = 1
            
            patch_mask_window_activated[x_start:x_end, y_start] = 1
            patch_mask_window_activated[x_start:x_end, y_end] = 1
            patch_mask_window_activated[x_start, y_start:y_end] = 1
            patch_mask_window_activated[x_end, y_start:y_end] = 1
            
            Activated_patch_frame[x_start:x_end, y_start:y_end] = pixelcolor_heatmap


    # Apply the mask to the original image
    filtered_event_frame = frame * patch_mask
    filtered_event_frame_with_window = frame + patch_mask_window_activated
    filtered_event_frame_with_window_activated = filtered_event_frame + patch_mask_window_activated

    colored_image = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 3), dtype=np.uint8)
    colored_image = Activated_patch_frame * frame_rgb
    
    # Display the original and the filtered images
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0][0].imshow(frame, cmap='gray')
    axes[0][0].set_title('Original event visualization')
    axes[0][0].axis('off')

    axes[0][1].imshow(filtered_event_frame, cmap='gray')
    axes[0][1].set_title('Activated event patch')
    axes[0][1].axis('off')

    axes[1][0].imshow(filtered_event_frame_with_window, cmap='gray')
    axes[1][0].set_title('Event patch with patch window')
    axes[1][0].axis('off')

    axes[1][1].imshow(filtered_event_frame_with_window_activated, cmap='gray')
    axes[1][1].set_title('Activated event patch with patch window')
    axes[1][1].axis('off')  

    axes[0][2].imshow(colored_image, cmap='gray')
    axes[0][2].set_title('Activated event patch heatmap')
    axes[0][2].axis('off')

    axes[1][2].imshow(Activated_patch_frame, cmap='gray')
    axes[1][2].set_title('Activated patch mask')
    axes[1][2].axis('off')

    plt_image_filename = activated_patch_mask_path + '/patch_'+str(patch_num)+'_ts_'+ str(i) + '_frame_' + str(j)+ '.png'
    print("Saved: ", plt_image_filename)
    plt.savefig(plt_image_filename, bbox_inches='tight', pad_inches=0)
    #plt.show()


# def target_frame_from_original_event_file(test_fn, path_dataset, result_path, activated_patch_mask_path, heap_map_path):
    
#     x, y, t, p = ReadAedat(test_fn, path_dataset, result_path, "SL_Animals_dvs")

#     # user12_indoor_S3_1.pckl
#     # startTime_ev = 352049
#     # endTime_ev = 461999
    
#     # user30_imse_S1_1.pckl
#     startTime_ev = 858695
#     endTime_ev = 965190
    
#     select_x = x[startTime_ev:endTime_ev]
#     select_y = y[startTime_ev:endTime_ev]
#     select_t = t[startTime_ev:endTime_ev]
#     select_p = p[startTime_ev:endTime_ev]    

    
#     total_time = select_t[-1] - select_t[0] #3048685 

#     time_length = 33333
#     start_index = 0
#     end_index = 0
#     frame_counter = 0
#     frame_time_start = select_t[start_index]
    
#     while(frame_time_start < select_t[-1]):
#         frame_time_end = frame_time_start + time_length
#         end_time_index_in_t = np.where(select_t<frame_time_end)[0][-1]
#         end_index = end_time_index_in_t

#         frame_x = select_x[start_index:end_index]
#         frame_y = select_y[start_index:end_index]
#         frame_t = select_t[start_index:end_index]
#         frame_p = select_p[start_index:end_index]

#         # Create an empty image with all zeros
#         sample_image = np.random.randint(0, 256, (128, 128), dtype=np.uint32)
#         frame = np.zeros_like(sample_image)
#         frame_rgb = np.zeros((128, 128, 3), dtype=np.uint8)

#         # Set the specified coordinates to 1
#         for x, y in zip(frame_x, frame_y):
#             frame[x, y] = 1
#             frame_rgb[x, y] = [1,1,1]

#         # Display the image
#         plt.imshow(frame, cmap='gray')
#         plt.axis('off')
        
#         # Save the image using plt.savefig
#         if os.path.exists(result_path + '/event_visual/') == False: os.mkdir(result_path + '/event_visual/')
#         plt_image_filename = result_path + '/event_visual/' + str(frame_counter)+ '.png'
#         plt.savefig(plt_image_filename, bbox_inches='tight', pad_inches=0)
#         plt.show()

#         activate_by_patch_mask(frame, frame_rgb, frame_counter, activated_patch_mask_path, heap_map_path)
        

        

#         frame_time_start = frame_time_end
#         start_index = end_index
#         frame_counter = frame_counter + 1

def total_events_chunk_visualization(test_fn, total_events_chunk_to_visual_path, total_events_chunk_to_visual_frame_path, activated_patch_mask_path, heap_map_path):

    total_events_chunk_to_visual = pickle.load(open(total_events_chunk_to_visual_path, 'rb'))

    for i in range(len(total_events_chunk_to_visual)):
        
        events_chunk = total_events_chunk_to_visual[i]
        
        for j in range(events_chunk.shape[0]):
            sample_image = np.random.randint(0, 256, (128, 128), dtype=np.uint32)
            frame = np.zeros_like(sample_image)
            events_to_visual_frame_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
            
            events = events_chunk[j].todense()
            events_x = np.where(events!=[0,0])[0]
            events_y = np.where(events!=[0,0])[1]
            events_to_visual_frame = np.zeros_like(frame)
            events_to_visual_frame[events_x+10, events_y+10]=1
            events_to_visual_frame_rgb[events_x+10, events_y+10] = [1,1,1]

            activate_by_patch_mask(events_to_visual_frame, events_to_visual_frame_rgb, i, j, activated_patch_mask_path, heap_map_path)


if __name__ == '__main__': 
    
    # test file name
    test_fn = 'user30_imse_S1_1' # default: 'user30_imse_S1_1'; option: 'speed_user30_imse_layer1_S1_1'
    test_struct_model = 'mevt_SME' # default: 'mevt_SME';  option: 'mevt_OF', 'evt'

    # SL_Animals
    path_dataset = './visual/Patch_visualization/'
    
    result_path = './visual/Patch_visualization/' + test_fn + '/'
    if os.path.exists(result_path) == False: os.mkdir(result_path)
    
    activated_patch_mask_path = result_path +  'activated_patch_visual_' + test_struct_model + '/'
    if os.path.exists(activated_patch_mask_path) == False: os.mkdir(activated_patch_mask_path)
    
    if os.path.exists('./visual/Patch_visualization/patch_heatmap'):
        shutil.move('./visual/Patch_visualization/patch_heatmap', activated_patch_mask_path )
    if os.path.exists('./visual/Patch_visualization/patch_mask_visual'):
        shutil.move('./visual/Patch_visualization/patch_mask_visual', activated_patch_mask_path )

    heap_map_path = activated_patch_mask_path + '/patch_heatmap/'
    if os.path.exists(heap_map_path) == False: os.mkdir(heap_map_path)

    #target_frame_from_original_event_file(test_fn, path_dataset, result_path, activated_patch_mask_path, heap_map_path)
    
    total_events_chunk_to_visual_frame_path = result_path + "/total_events_chunk_to_visual_frame/"
    if os.path.exists(total_events_chunk_to_visual_frame_path) == False: os.mkdir(total_events_chunk_to_visual_frame_path)

    total_events_chunk_to_visual_path = path_dataset + 'total_events_chunk_to_visual.pckl'
    total_events_chunk_visualization(test_fn, total_events_chunk_to_visual_path, total_events_chunk_to_visual_frame_path, activated_patch_mask_path, heap_map_path)
    

