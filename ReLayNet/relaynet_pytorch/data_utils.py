"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import h5py
from pathlib import Path


class ImdbData(data.Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        weight = self.w[index]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        return img, label, weight

    def __len__(self):
        return len(self.y)


def get_imdb_data(dir_path = None, suffix = '', row_slice=("start","end"), col_slice=("start","end")): #, row_upper_limit=0, column_lower_limit=0): # ECG TODO update default values 
# row_slice and col_slice determine slicing below
    
    
    ### ECG Edits
    if dir_path != None:
        path_data = str(dir_path / f"data{suffix}.h5")
        path_labels = str(dir_path / f"labels{suffix}.h5")
        path_set = str(dir_path / f"set{suffix}.h5")
    else:
        path_data = "datasets/Data.h5"
        path_labels = "datasets/label.h5"
        path_set = 'datasets/set.h5'
    ###
    
    # TODO: Need to change later
    # NumClass = 9 # original value
    NumClass = 10

    # Load DATA
    print("Loading data.h5")
    Data = h5py.File(path_data, 'r')
    #Data = h5py.File('datasets/Data.h5', 'r')
    a_group_key = list(Data.keys())[0]
    Data = list(Data[a_group_key])
    Data = np.squeeze(np.asarray(Data)) ## removes channel dimension (num_images, rows, cols)
    #### ECG modifications
    #print(f"data shape: {Data.shape}")
    ####
    
    print("Loading label.h5")
    Label = h5py.File(path_labels, 'r')
    #Label = h5py.File('datasets/label.h5', 'r')
    a_group_key = list(Label.keys())[0]
    Label = list(Label[a_group_key])
    Label = np.squeeze(np.asarray(Label))
    
    #### ECG modifications
    #print(f"label shape: {Label.shape}")
    ####
    
    print("Loading set.h5")
    set = h5py.File(path_set, 'r')
    #set = h5py.File('datasets/set.h5', 'r')
    a_group_key = list(set.keys())[0]
    set = list(set[a_group_key])
    set = np.squeeze(np.asarray(set))
    
    ## FORMAT DATA
    
    #row slicing
    img_rows, img_cols = Data[0,:,:].shape # image limits
    
    r_start, r_stop = row_slice
    r_start = 0 if r_start == "start" else r_start
    r_stop = img_rows if r_stop == "end" else r_stop
    
    #col slicing
    c_start, c_stop = col_slice
    c_start = 0 if c_start == "start" else c_start
    c_stop = img_cols if c_stop == "end" else c_stop
    
    print(f"slicing: ({r_start}:{r_stop}, {c_start}:{c_stop})")
    
    sz = Data.shape
    Data = Data.reshape([sz[0], 1, sz[1], sz[2]])
    Data = Data[:, :, r_start:r_stop, c_start:c_stop] # (num_images, channel, rows, cols) # this slicing creates 512x512 image
                                 #originally [:, :, 61:573, :]
                                 #our Duke dataset [:, :,130:642 , :] 
    #get labels and weights
    #512x512 image
    weights = Label[:, 1, r_start:r_stop, c_start:c_stop] # May's values for OCT data [:, 1, 0:256, 100:] array of [256,3000]
    Label = Label[:, 0, r_start:r_stop, c_start:c_stop]  # modify img dimensions
    sz = Label.shape
    Label = Label.reshape([sz[0], 1, sz[1], sz[2]])
    weights = weights.reshape([sz[0], 1, sz[1], sz[2]])
    
    train_id = set == 1
    test_id = set == 3
    
    Tr_Dat = Data[train_id, :, :, :]                
    Tr_Label = np.squeeze(Label[train_id, :, :, :]) - 1 # Index from [0-(NumClass-1)]
    Tr_weights = weights[train_id, :, :, :]             
    Tr_weights = np.tile(Tr_weights, [1, NumClass, 1, 1])

    Te_Dat = Data[test_id, :, :, :]
    Te_Label = np.squeeze(Label[test_id, :, :, :]) - 1  # Index from [0-(NumClass-1)]
    Te_weights = weights[test_id, :, :, :]              
    Te_weights = np.tile(Te_weights, [1, NumClass, 1, 1])



    return (ImdbData(Tr_Dat, Tr_Label, Tr_weights),
            ImdbData(Te_Dat, Te_Label, Te_weights))
