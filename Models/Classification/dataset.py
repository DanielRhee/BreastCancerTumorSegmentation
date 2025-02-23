import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class ClassifyDS(Dataset):
    def __init__(self, imgPath, labels, transform=None):
        self.dataframe = pd.read_csv(labels)
        self.imgPath = imgPath

        #indices = self.dataframe["CaseID"].tolist()

        #res = []
        #for i in indices:
        #    res.append(Image.open(imgPath + "/case" + str(i).zfill(3) + ".png"))
        #self.dataframe["img"] = res

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        
        label = self.dataframe.iloc[index, 1]
        #print(label, self.dataframe.iloc[index, 0])

        image = Image.open(self.imgPath + "/case" + str(self.dataframe.iloc[index, 0]).zfill(3) + ".png")

        if self.transform:
            image = self.transform(image)

        print(label)

        return image, label