"""Lesion data for aphasia patients."""
import os
import numpy as np
import torch
import nibabel as nib
import cv2
import torch.utils.data as data
from glob import glob
from sklearn.model_selection import train_test_split

__all__ = ['LesionSegmentation']

class LesionSegmentation(data.Dataset):
    """Lesion Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to Lesion folder. Default is './datasets/lesion/2024-04-01-MedSAM-on-Laptop/datasets/ARC_Brain_Tumor/ds004884'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = LesionSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    
    def __init__(self, root='../datasets/lesion/2024-04-01-MedSAM-on-Laptop/datasets/ARC_Brain_Tumor/ds004884', split='train', transform=None):
        super(LesionSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.img_paths, self.mask_paths = _get_lesion_pairs(self.root)
        assert len(self.img_paths) == len(self.mask_paths), "Mismatch between images and masks"
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        self.valid_classes = [1]

        def __len__(self):
            return len(self.images)

        @property
        def num_class(self):
            """Number of categories."""
            return self.NUM_CLASS

 
    def __getitem__(self, index):
        img = nib.load(self.images[index]).get_fdata()
        mask = nib.load(self.mask_paths[index]).get_fdata()

        X_train, X_test, y_train, y_test  = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        
        # synchrosized transform
        if self.split == 'train':
            img, mask = self._sync_transform(X_train, y_train)
        elif self.split == 'val':
            img, mask = self._val_sync_transform(X_val, y_val)
        else:
            assert self.split == 'test'
            img, mask = self._img_transform(X_test), self._mask_transform(y_test)


def _get_lesion_pairs(root):
    """Get image and mask pairs for the Lesion dataset."""
    img_paths = []      ## sub-M2002_ses-1441_acq-tse3_run-4_T2w.nii.gz
    mask_paths = []     ## sub-M2002_ses-1441_acq-tse3_run-4_T2w_desc-lesion_mask.nii.gz
    # Load images and masks from the dataset directory
    for mask_path in glob(os.path.join(root, 'derivatives/lesion_masks', 'sub-*/ses-*/anat/*.nii.gz')):
        sub = mask_path.split('/')[-4]
        ses = mask_path.split('/')[-3]
        filename = mask_path.split('/')[-1].replace('_desc-lesion_mask.nii.gz', '.nii.gz')
        img_path = os.path.join(root, sub, ses, filename)
        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            img_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print('cannot find the mask or image:', img_path, mask_path)
    print('Found {} images in the folder {}'.format(len(img_paths)))
    return img_paths, mask_paths

def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

brain = nb.load('/data/Projects/cvpr-sam-on-laptop-2024/test_demo/imgs/sub-M2002_ses-a1440_T2w.nii').get_fdata()
print("len(brain)", brain.shape, np.max(brain), np.min(brain))

nifti_slice = brain[:,:,90]
print("nifti_slice", nifti_slice.shape, nifti_slice.dtype, np.max(nifti_slice), np.min(nifti_slice))
image_data_pre = (nifti_slice - np.min(nifti_slice))/(np.max(nifti_slice)-np.min(nifti_slice)) * 255
print("image_data_pre", image_data_pre.shape, np.max(image_data_pre), np.min(image_data_pre))
image_data_pre = np.uint8(image_data_pre)
img_3c = np.stack((image_data_pre,)*3, axis=-1)