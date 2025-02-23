import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class ClassifyDS(Dataset):
    def __init__(self, imgPath, labels, transform=None):
        self.dataframe = pd.read_csv(labels)
        self.dataframe['Classification'] = self.dataframe['Classification'].replace({'benign': 0, 'malignant': 1, 'normal': 2})
        #print(len(self.dataframe))
        self.imgPath = imgPath
        
        # Create a mapping of unique labels to numbers
        unique_labels = self.dataframe.iloc[:, 1].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Create reverse mapping to print the label as a string
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
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

        # Print the class associated with the index
        #print(f"Index {index} is associated with class '{self.idx_to_label[label.item()]}'")
        
        return image, label
