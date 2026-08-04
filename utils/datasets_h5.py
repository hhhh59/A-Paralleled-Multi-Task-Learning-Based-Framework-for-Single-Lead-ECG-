import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import wfdb
from tqdm import tqdm
from wfdb import processing
import pdb
# import neurokit2 as nk
import pandas as pd
import h5py
import h5pickle

class CustomDataset(Dataset):
    def __init__(self, data_path: str = ""):
        self.file = h5pickle.File(data_path, 'r',skip_cache=False)
        # self.data = self.file['signals'][:int(11322*0.8),:,:]
        # self.signals = self.data[:int(11322*0.8),:,]
        self.data = self.file['signals'][:,:,:]
        self.signals = self.data[:,:,]
        print(self.data.shape)
# train [0:7925]
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # print(idx)
        return np.array(self.signals[idx])[None,:,:], np.array(idx)
        # x, y = np.array(self.signals[idx])[None,:,:], np.array(idx)


class CustomDataset_Single_lead(Dataset):
    def __init__(self, data_path: str = ""):
        self.file = h5pickle.File(data_path, 'r', skip_cache=False)
        data = self.file['dataset']
        data = np.array(data).transpose((1, 0)).reshape((data.shape[1], 1, 1001))
        self.data = data[:,:,:]
        self.signals = self.data
        del data
# train: 95104
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.signals[idx])[None, :, :], np.array(idx)

class CustomDataset_Single_lead_Denoise(Dataset):
    def __init__(self, data_path: str = ""):
        self.file = h5pickle.File(data_path, 'r', skip_cache=False)
        data = self.file['dataset']
        data = np.array(data).transpose((2, 0, 1))
        self.data = data[:int(11322*12*0.7),:,]
        del data
        self.signals = self.data[:int(11322*12*0.7),:,]
# train: 95104
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.signals[idx])[None, :, :], np.array(idx)

class CustomDataset_Denoise(Dataset):
    def __init__(self, data_path: str = ""):
        self.file = h5pickle.File(data_path, 'r', skip_cache=False)

        self.data = self.file['signals']
        self.signals = self.data[:]

        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.signals[idx])[None, :, :2000], np.array(idx)


# file = h5pickle.File('ptb_xl/standard_scaler.pkl', 'r',skip_cache=False)

# import pickle
# with open('ptb_xl/' + 'standard_scaler.pkl', 'wb') as ss_file:
#     pickle.dump('ss', ss_file)

#
# # 替换为你的 .pkl 文件路径
# file_path = 'ptb_xl/standard_scaler.pkl'
#
# # 打开 .pkl 文件并加载数据
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # 打印或处理加载的数据
# print(data)


