import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CancerDataset(Dataset):
    def __init__(self, datasetDir, transform=None):
        self.image_dir = datasetDir
        self.mask_dir = datasetDir
        self.transform = transform
        self.images = [file for file in os.listdir(datasetDir) if file.lower().endswith('.png')]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_tumor.png"))
        
        image = np.array(Image.open(img_path).convert("RGB"))
        masks = np.array(Image.open(mask_path).convert)

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=masks)
            image = augmentations["image"]
            masks = augmentations["masks"]

        return image, masks