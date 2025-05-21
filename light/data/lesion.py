"""Lesion data for aphasia patients."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
join = os.path.join

__all__ = ['NpzDataset']

NUM_CLASS = 1
#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.ori_imgs = np.vstack([d['imgs'] for d in self.npz_data])
        # print(f"{self.ori_imgs.shape=}, {self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        gt2D = self.ori_gts[index]
        img2D = self.ori_imgs[index]

        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(gt2D[None, :,:]).long(), torch.tensor(img2D[None, :,:]).long()