import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class ClassifyDS(Dataset):
    def __init__(self, imgPath, labels, transform=None):
        self.dataframe = pd.read_csv(labels)
        print(len(self.dataframe))
        self.imgPath = imgPath
        
        # Create a mapping of unique labels to numbers
        unique_labels = self.dataframe.iloc[:, 1].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Get the string label
        label_str = self.dataframe.iloc[index, 1]
        # Convert to numerical label
        label = self.label_to_idx[label_str]
        
        # Open image and convert to RGB
        image = Image.open(self.imgPath + "/case" + str(self.dataframe.iloc[index, 0]).zfill(3) + ".png").convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label