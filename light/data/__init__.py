"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes import CitySegmentation
from .lesion import NpzDataset

datasets = {
    'citys': CitySegmentation,
    'lesion': NpzDataset, 
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)