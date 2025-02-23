import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class ClassifyCNN(nn.Module):
    def __init__(self, num_classes):
        super(ClassifyCNN, self).__init__()
        # Input is 128x128x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)      
        
        self.fc1 = nn.Linear(32 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))   
        x = self.pool(self.relu(self.conv2(x)))   
        x = x.view(-1, 32 * 32 * 32)             
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x