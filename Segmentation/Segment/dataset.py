import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class CancerDataset(Dataset):
    def __init__(self, datasetDir, transform=None):
        self.image_dir = datasetDir
        self.mask_dir = datasetDir
        self.transform = transform
        self.images = [file for file in os.listdir(datasetDir) 
                      if file.lower().endswith('.png') and not file.endswith('_tumor.png')]
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_tumor.png"))
        
        # Load grayscale image and repeat it 3 times to create 3 channels
        image = np.array(Image.open(img_path).convert("L"))
        image = np.stack([image] * 3, axis=-1)  # Create 3 channels (H, W, 3)
        
        # Load and process mask
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 128).astype(np.float32)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]  # This will be in format (C, H, W)
            mask = augmentations["mask"]    # This will be in format (H, W)
            
        # Ensure mask is in the correct format (C, H, W)
        mask = mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).unsqueeze(0)

        return image, mask
    
    def __len__(self):
        return len(self.images)