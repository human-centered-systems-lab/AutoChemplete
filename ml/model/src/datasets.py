import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class SmilesDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, base_file_name, split, transform=None, grayscale=False):
        """
        :param data_folder: folder where data files are stored
        :param base_file_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        if self.split in {'TRAIN', 'VAL'}:
            self.h = h5py.File(data_folder / f"{self.split}_IMAGES_{base_file_name}.hdf5", 'r')
            # Load encoded sequences (completely into memory)
            with open(data_folder / f"{self.split}_SMILES_SEQUENCES_{base_file_name}.json", 'r') as j:
                self.sequences = json.load(j)

            # Load sequence lengths (completely into memory)
            with open(data_folder / f"{self.split}_SMILES_SEQUENCE_LENS_{base_file_name}.json", 'r') as j:
                self.sequence_lens = json.load(j)

        else:  # self.split in {'TEST'}
            self.h = h5py.File(data_folder / "TEST_LG_IMAGES.hdf5", 'r')

        self.imgs = self.h['images']

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.dataset_size = self.imgs.shape[0]
        if grayscale:
            self.grayscale_transform = transforms.Grayscale(3)
        else:
            self.grayscale_transform = None

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        if self.grayscale_transform is not None:
            img = self.grayscale_transform(img)

        if self.split in {'TRAIN', 'VAL'}:
            sequence = torch.LongTensor(self.sequences[i])
            sequence_len = torch.LongTensor([self.sequence_lens[i]])
            return img, sequence, sequence_len
        else:
            return img

    def __len__(self):
        return self.dataset_size


class PNGSmileDataset(SmilesDataset):

    def __init__(self, data_folder, base_file_name, split, transform=None, grayscale=False):
        """
        :param data_folder: folder where data files are stored
        :param base_file_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # instead of using hdf5 (list of preprocessed imgs), we store a list of images in pkl

        # Open hdf5 file where images are stored
        if self.split in {'TRAIN', 'VAL'}:
            # self.h = h5py.File(data_folder/ f"{self.split}_IMAGES_{base_file_name}.hdf5", 'r')
            self.img_list = torch.load(data_folder / f"{self.split}_IMAGES_{base_file_name}.pt")

            # Load encoded sequences (completely into memory)
            with open(data_folder / f"{self.split}_SMILES_SEQUENCES_{base_file_name}.json", 'r') as j:
                self.sequences = json.load(j)

            # Load sequence lengths (completely into memory)
            with open(data_folder / f"{self.split}_SMILES_SEQUENCE_LENS_{base_file_name}.json", 'r') as j:
                self.sequence_lens = json.load(j)



        else:  # self.split in {'TEST'}
            self.h = h5py.File(data_folder / "TEST_LG_IMAGES.hdf5", 'r')

        # self.imgs = self.h['images']

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.dataset_size = len(self.img_list)

        if grayscale:
            self.grayscale_transform = transforms.Grayscale(3)
        else:
            self.grayscale_transform = None

    def __getitem__(self, i):
        img_path = self.img_list[i]

        img = Image.open(img_path)
        img = self.png_to_tensor(img)
        # img = img.resize((256, 256))
        # img = np.array(img)
        # # img:(width， height， channel)
        # img = np.rollaxis(img, 2, 0)
        #
        # img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)


        if self.grayscale_transform is not None:
            img = self.grayscale_transform(img)
        # print(img.size())

        if self.split in {'TRAIN', 'VAL'}:
            sequence = torch.LongTensor(self.sequences[i])
            sequence_len = torch.LongTensor([self.sequence_lens[i]])
            return img, sequence, sequence_len
        else:
            return img



    def png_to_tensor(self, img: Image):
        """
        convert png format image to torch tensor with resizing and value rescaling
        :param img: .png file
        :return: tensor data of float type
        """
        img = img.resize((256,256))
        img = np.array(img)

        # what is the dimension of img?
        # (
        #print(img.ndim)
        if img.ndim == 3:
            img = np.moveaxis(img, 2, 0) # this function moves the final axis to the first
            # it means that the img can be [256, 256, 3]  -> [3, 256, 256] #[N C H W] is right
            # or [3, 256, 256] -> [256, 256, 3]] # [N H W C]
        else:
            # now with only grayscale your image is [256, 256] ->
            img = np.stack([img, img, img], 0)

        return torch.FloatTensor(img) / 255.


    def __len__(self):
        return self.dataset_size