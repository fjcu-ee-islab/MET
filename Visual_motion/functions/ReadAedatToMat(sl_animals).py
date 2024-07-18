import os
import struct
import numpy as np
from dv import AedatFile
os.chdir('..')

import aermanager
from aermanager.aerparser import load_events_from_file
from aermanager.parsers import parse_header_from_file, get_aer_events_from_file

from tqdm import tqdm
from scipy.io import savemat

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

        
def ReadAedatToMat(path_dataset, result_path, dataset_name):
    files = os.listdir(path_dataset)
    x, y, t, p = [], [], [], []

    for events_file in tqdm(files):
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
    
        outputFileName = result_path + events_file[:-7] + ".mat"
        os.makedirs(result_path, exist_ok=True)
        savemat(outputFileName, events)
    
    
# Source data folder

# SL_Animals
'''
path_dataset = '/home/eden/Desktop/sda1/EventTransformer/datasets/SL_Animals/allusers_aedat/'

result_path = 
ReadAedatToMat(path_dataset, result_path, "SL_Animals")
'''

# Fall Detection
'''
path_dataset = '/home/eden/Desktop/sda1/Spiking_Motion/Data/Fall Detection/aedat4 version/'

# For ".aedat" file, use events_file[:-6] to get the file name, like "SL_Animals_dvs" datasets.
# outputFileName = "/home/eden/Desktop/sda1/Spiking_Motion/Data/Fall Detection/mat version files/" + events_file[:-6] + ".mat"

# But for ".aedat4" file, use events_file[:-7] to get the file name, like "Fall Detection Dataset" datasets.

outputFileName = "/home/eden/Desktop/sda1/Spiking_Motion/Data/Fall Detection/mat version files/" + events_file[:-7] + ".mat"

result_path = "/home/eden/Desktop/sda1/Spiking_Motion/Data/Fall Detection/mat version files/"
ReadAedatToMat(path_dataset, result_path, "Fall Detection")
'''

# UCF11 (240x240)
'''
path_dataset = '/home/eden/Desktop/sda1/Downloads/UCF11/AEDAT4.0/'
result_path = '/home/eden/Desktop/sda1/Spiking_Motion/Data/UCF11/'
setsName = []
setsName = os.listdir(path_dataset)
for indexNum in range(len(setsName)):
    ReadAedatToMat(path_dataset+setsName[indexNum]+'/',result_path+setsName[indexNum]+'/', "UCF11")
''' 
# IITM (128x128)
path_dataset = '/home/eden/Desktop/sda1/Downloads/IITM/AEDAT4.0/'
result_path = '/home/eden/Desktop/sda1/Spiking_Motion/Data/IITM/'
setsName = []
setsName = os.listdir(path_dataset)
for indexNum in range(len(setsName)):
    ReadAedatToMat(path_dataset+setsName[indexNum]+'/',result_path+setsName[indexNum]+'/', "IITM")
