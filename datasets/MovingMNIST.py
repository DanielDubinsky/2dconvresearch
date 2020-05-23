import torch
import numpy as np
from torch.utils.data import Dataset


class MovingMNIST(Dataset):
    def __init__(self, np_arr_path: str, train: bool, transform=None, **kwargs):
        self.transform = transform

        np_arr = np.load(np_arr_path)

        self.images = torch.from_numpy(np_arr)

        num_sequences = self.images.size(1)
        num_train = int(0.8 * num_sequences)
        if train:
            self.images = self.images[:, num_train:]
        else:
            self.images = self.images[:, : num_train]

        self.images = self.images.unsqueeze(2)

    def __len__(self):
        return self.images.size(1)

    def __getitem__(self, idx):
        images = self.images[:, idx]
        sample = (images[:10], images[10])
        if self.transform:
            sample = (self.transform(sample[0]), self.transform(sample[1]))

        return sample
