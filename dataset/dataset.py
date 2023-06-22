import os

import torch.nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UNetDataset(Dataset):
    def __init__(self, l, path, train):
        super(UNetDataset, self).__init__()
        self.len = l
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train = train
        self.data_path = path

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        path_to_image = os.path.join(self.data_path, 'volume', f"{item if self.train else item + 40}.png")
        path_to_label = os.path.join(self.data_path, 'segmentations', f"{item if self.train else item + 40}.png")
        img = Image.open(path_to_image)
        img = self.transform(img)
        label = Image.open(path_to_label)
        label = self.transform(label)
        label = torch.where(label > 0.5, 1, 0)
        return img, label


if __name__ == "__main__":
    img = Image.open('../2d/volume/0.png')
    m = UNetDataset(40, '../2d', False)
    img, label = m[0]
    img.show()
