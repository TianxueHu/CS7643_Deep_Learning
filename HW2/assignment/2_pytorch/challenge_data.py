# NOTE: This is adapted from the torchvision datasets for
# CIFAR10 and CIFAR100, which can be found at
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py

from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import urllib.request

class ChallengeData(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``test_images.npy`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = "https://s3.amazonaws.com/cs7643-fall2019/test_images.npy"
    filename = "test_images.npy"


    def __init__(self, root,
                 transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        
        if download:
            self.download()

        # now load the picked numpy arrays
        file = os.path.join(self.root, self.filename)
        self.test_data = np.load(file)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.test_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.test_data)

    def download(self):
        root = self.root
        if not os.path.exists(os.path.join(root, self.filename)):
            print("Downloading data...")
            urllib.request.urlretrieve(self.url, os.path.join(root, self.filename))
            print("Download complete")

