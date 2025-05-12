"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes import CitySegmentation
from .lesion import LesionSegmentation

datasets = {
    'citys': CitySegmentation,
    'lesion': LesionSegmentation, 
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)