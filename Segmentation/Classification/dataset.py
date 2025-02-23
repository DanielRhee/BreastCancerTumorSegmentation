import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class ClassifyDS(Dataset):
    def __init__(self, imgPath, labels, transform=None):
        self.data_frame = pd.read_csv(labels)
        self.imgPath = imgPath
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.imgPath, self.data_frame.iloc[index, 0])
        image = Image.open(img_name).convert("L")
        label = self.data_frame.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        return image, label