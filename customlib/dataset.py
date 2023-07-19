import os
import numpy as np
import zipfile
import PIL.Image
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import pyspng


class sinogramDataset(Dataset):
    def __init__(self,
                 path: str,
                 ):
        self._path = path
        self._all_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self._path)
            for root, _dirs, files in os.walk(self._path)
            for fname in files
        }

        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == '.npy')
        if len(self._image_fnames) == 0:
            raise IOError(f'No image files found in the specified path : {self._path}')

        self._name = os.path.splitext(os.path.basename(self._path))[0]
        self._img_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

    def num_channels(self):
        return self._img_shape[1]

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def shape(self):
        return self._img_shape

    def __len__(self):
        return len(self._image_fnames)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image = np.load(os.path.join(self._path, fname))
        image = torchvision.transforms.functional.to_tensor(
            image,
        )
        return image

    def __getitem__(self, item):
        image = self._load_raw_image(item)
        assert list(image.shape) == self._img_shape[1:4], print(image.shape, self._img_shape[1:4])
        return image


def set_dataset(config):
    print(f"Preparing dataset")
    print(f"Dataset [{config.dataname}] at {config.datadir}")
    basedir = os.path.join(config.datadir, config.dataname)
    __batchsize__ = config.batchsize
    print(f"Batch size: {__batchsize__}")

    # Dataset for training
    __traindir__ = os.path.join(basedir, 'train')
    ds = sinogramDataset(__traindir__)
    print(f"Number of training samples: {len(ds)}")
    print(f"Size of sinogram: {ds.shape()}")
    assert ds.shape()[3] == config.num_det and ds.shape()[2] == config.view, f"Error! the shape of sinogram data ({ds.shape}) is different from the setting ({config.num_det}, {config.view_num})"

    # Dataset for validation
    __valdir__ = os.path.join(basedir, 'val')
    ds_v = sinogramDataset(__valdir__)
    print(f"Number of valdation samples: {len(ds_v)}")

    return DataLoader(
        dataset=ds,
        batch_size=__batchsize__,
        shuffle=True,
        num_workers=config.numworkers,
        pin_memory=True
    ), DataLoader(
        dataset=ds_v,
        batch_size=config.valbatchsize,
        shuffle=False,
        num_workers=config.numworkers,
        pin_memory=True
    ), ds.num_channels()
