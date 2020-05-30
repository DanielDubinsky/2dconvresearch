import torch
import numpy as np
from torch.utils.data import Dataset


class MovingMNIST(Dataset):
    def __init__(self, np_arr_path: str, train: bool, transform=None, **kwargs):
        self.transform = transform

        np_arr = np.load(np_arr_path)

        self.images = torch.from_numpy(np_arr)
        self.images = self.images.unsqueeze(2)

        train_ratio = 0.9

        num_sequences = self.images.size(1)
        num_train = int(train_ratio * num_sequences)

        self.train_images = self.images[:, : num_train]
        self.test_images = self.images[:, num_train:]
        if train:
            self.images = self.train_images
        else:
            self.images = self.test_images

        if train_ratio == 0.9:
            self.mean = 12.567464828491211
            self.std = 51.07254409790039
        else:
            self.mean = torch.mean(self.train_images.to(dtype=torch.float32))
            self.std = torch.std(self.train_images.to(dtype=torch.float32))

    def __len__(self):
        return self.images.size(1)

    def __getitem__(self, idx):
        images = self.images[:, idx]
        sample = (images[:10], images[10])
        if self.transform:
            sample = (self.transform(sample[0]), self.transform(sample[1]))

        return sample
