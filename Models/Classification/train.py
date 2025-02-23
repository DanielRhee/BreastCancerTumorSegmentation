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

imgPath = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Raw')
labels = os.path.join(os.path.dirname(os.getcwd()), 'Data') + "/Classification.csv"

class_counts = pd.read_csv(labels)['Classification'].value_counts()  
class_weights = torch.tensor([1.0 / class_counts['benign'], 1.0 / class_counts['malignant'], 1.0 / class_counts['normal']], dtype=torch.float32).to(DEVICE)

# Pass the class_weights to the criterion
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
])



dataset = dataset.ClassifyDS(imgPath= imgPath, labels=labels , transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Print shapes for debugging
        #print(f"Image batch shape: {images.shape}")
        #print(f"Labels shape: {labels.shape}")
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Print output shape
        #print(f"Outputs shape: {outputs.shape}")
        
        loss = criterion(outputs, labels)
        loss.backward()


        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

checkpoint = {jjn
    "state_dict": model.state_dict(),
     "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, "classify.pth.tar")