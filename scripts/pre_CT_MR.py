#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 6 19:25:36 2023

convert nonCT nii image to npz files, including input image, image embeddings, and ground truth
Compared to pre_CT.py, the main difference is the image intensity normalization method (see line 66-72)

@author: jma
"""
#%% import packages
import numpy as np
import SimpleITK as sitk
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from torch.nn import functional as F
from glob import glob
# set up the parser
parser = argparse.ArgumentParser(description='preprocess non-CT images')
parser.add_argument('-i', '--nii_path', type=str, default='./datasets/lesion_sample', help='path to the nii images')
parser.add_argument('-gt', '--gt_path', type=str, default='./datasets/lesion_sample/derivatives/lesion_masks', help='path to the ground truth',)
parser.add_argument('-o', '--npz_path', type=str, default='./npz_wholebbox_no_sam_preprocess', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--modality', type=str, default='MRI', help='modality')
parser.add_argument('--anatomy', type=str, default='lesion', help='anatomy')
parser.add_argument('--img_name_suffix', type=str, default='.nii', help='image name suffix')
parser.add_argument('--label_id', type=int, default=1, help='label id')
# seed
parser.add_argument('--seed', type=int, default=2023, help='random seed')
args = parser.parse_args()

prefix = args.modality + '_' + args.anatomy

names = []
for (root,dirs,files) in os.walk(args.gt_path):
    for file in files:
        if file.endswith('.nii.gz'):
            names.append(file)
    print('names', names)

names = sorted(names)

# def preprocessing function
def preprocess_nonct(gt_path, nii_path, label_id, image_size):
    """
    Return lists of resized 2d images and ground truth. For training, also return image embeddings
    """

    gt_sitk = sitk.ReadImage(gt_path)
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data = np.uint8(gt_data==label_id)

    if np.sum(gt_data)>0: # compact taregets are not considered, because they are easier to segment. The key challenge for small targets (e.g., tumors) is detection rather than segmentation
        imgs = []
        gts =  []
        assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
        img_sitk = sitk.ReadImage(nii_path)
        image_data = sitk.GetArrayFromImage(img_sitk)
        # print('image_encoder.img_size', image_data.shape, np.mean(image_data), "std=", np.std(image_data), "min=", np.min(image_data), "max=", np.max(image_data))

        # nii preprocess start
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = np.uint8(image_data_pre)
        
        index = image_data.shape[0]
        for i in range(index):
            gt_slice_i = gt_data[i,:,:]
            # print('gt_slice_i', gt_slice_i.shape)
            gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)

            if np.sum(gt_slice_i)>3:
                img_slice_i = image_data_pre[i,:,:]

                # convert to three channels
                img_3c = np.repeat(img_slice_i[:, :, None], 3, axis=-1)
                # resize img_slice_i to 256x256
                resize_img_skimg = transform.resize(img_3c, (image_size*4, image_size*4), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
 
                resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                    resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)
                # print(resize_img_skimg_01.shape, "pre mean=", np.mean(resize_img_skimg_01), "std=", np.std(resize_img_skimg_01), "max=", np.max(resize_img_skimg_01))

                assert len(resize_img_skimg_01.shape)==3 and resize_img_skimg_01.shape[2]==3, 'image should be 3 channels'
                # assert resize_img_skimg_01.shape[0]==gt_slice_i.shape[0] and resize_img_skimg_01.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size but got {} and {}'.format(resize_img_skimg_01.shape, gt_slice_i.shape)
                imgs.append(img_slice_i)
                # print('img_slice_i', img_slice_i.shape)
                # assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
                gts.append(gt_slice_i)

                # print(i, "input_image mean=", np.mean(img_slice_i), "std=", np.std(img_slice_i), "min=", np.min(img_slice_i), "max=", np.max(img_slice_i))
    return imgs, gts


#%% prepare the save path
save_path_tr = join(args.npz_path, prefix, 'train')
save_path_ts = join(args.npz_path, prefix, 'test')
os.makedirs(save_path_tr, exist_ok=True)
os.makedirs(save_path_ts, exist_ok=True)

for name in tqdm(names):
    image_name = name.replace('_desc-lesion_mask.nii.gz', '.nii.gz')
    print('image_name', image_name)
    gt_name = name 
    gt_dir = os.path.join(args.gt_path, name.split('_')[0], name.split('_')[1], "anat", name)
    print('gt_dir', gt_dir)
    img_dir = os.path.join(args.nii_path, name.split('_')[0], name.split('_')[1], "anat", image_name)
    imgs, gts = preprocess_nonct(gt_dir, img_dir, args.label_id, args.image_size)
    # break
    #%% save to npz file
    # stack the list to array
    if len(imgs)>1:
        imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0) # (n, 256, 256)
        print(name, 'imgs shape', imgs.shape, '\tgts shape', gts.shape)
        np.savez_compressed(join(save_path_tr, prefix + '_' + gt_name.split('.nii')[0]+'.npz'), imgs=imgs, gts=gts)
 