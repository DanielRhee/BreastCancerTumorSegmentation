import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CancerDataset(Dataset):
    def __init__(self, datasetDir, transform=None):
        self.image_dir = datasetDir
        self.mask_dir = datasetDir
        self.transform = transform
        self.images = [file for file in os.listdir(datasetDir) 
                      if file.lower().endswith('.png') and not file.endswith('_tumor.png')]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_tumor.png"))
        
        # Load grayscale image and repeat it 3 times to create 3 channels
        image = np.array(Image.open(img_path).convert("L"))
        image = np.stack([image] * 3, axis=-1)  # Create 3 channels
        
        # Load and process mask
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 128).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)    # Shape: (H, W, 1)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask