import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import dataset
from model import ClassifyCNN
from torch.utils.data import DataLoader

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = "cpu"
    print ("MPS device not found.")

num_classes = 3
model = ClassifyCNN(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
])


imgPath = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Raw')
labels = os.path.join(os.path.dirname(os.getcwd()), 'Data') + "/Classification.csv"
dataset = dataset.ClassifyDS(imgPath= imgPath, labels=labels , transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images = images.to(DEVICE)
        print(labels)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'classify.pth.tar')