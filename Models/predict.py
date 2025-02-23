import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Models.Segment.model import UNET
from Models.Classification.model import ClassifyCNN
import numpy as np
from PIL import Image
import os
from io import BytesIO
import matplotlib.pyplot as plt

class predictor:
    def __init__(self):
        
        if torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
        else:
            print("cpu time :(")
            self.DEVICE = torch.device("cpu")
        self.model = self.loadModel()
        self.classifyModel = self.loadClassification()


    def predict(self, image):
        img = np.array(image) 

        mask = self.predict_mask(self.model, img, device=self.DEVICE)
        classOfTumor = self.getClassification(self.classifyModel, img)

        overlay = self.create_overlay(img, mask)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Predicted Mask')
        ax2.axis('off')

        ax3.imshow(overlay)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        # Convert buffer to PIL Image
        buf.seek(0)
        image = Image.open(buf)

        

        return image, classOfTumor

    def loadModel(self):
        model = UNET(in_channels=3, out_channels=1).to(self.DEVICE)
        model.load_state_dict(torch.load("Models/model.pth.tar", map_location=self.DEVICE)["state_dict"])
        return model
    
    def loadClassification(self):
        model = ClassifyCNN(num_classes=3).to(self.DEVICE)
        model.load_state_dict(torch.load("classify.pth.tar", map_location=self.DEVICE)["state_dict"])
        model.eval()
        return model

    def preprocess_image(self, image):
        transform = A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Update this if needed
            ToTensorV2()
        ])
        augmented = transform(image=image)
        return augmented["image"].unsqueeze(0).to(self.DEVICE)

    def getClassification(self, model, image):
        image_tensor = self.preprocess_image(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item() 
            probabilities = torch.softmax(outputs, dim=1)  
        return predicted_class, probabilities

    
    def create_overlay(self, original_image, mask, alpha=0.5, color=[1, 0, 0]):
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        colored_mask = np.zeros_like(original_image)
        for i in range(3):
            colored_mask[:, :, i] = mask * color[i]
        
        overlay = original_image.copy()
        mask_3d = np.stack([mask] * 3, axis=-1)
        overlay[mask_3d > 0] = (1 - alpha) * original_image[mask_3d > 0] + alpha * colored_mask[mask_3d > 0]
        
        return overlay

    def predict_mask(self, model, image_raw, device="cpu"):
        self.model.eval()
        originalShape = image_raw.shape

        transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

        image_tensor = transform(image=image_raw)["image"]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = torch.sigmoid(self.model(image_tensor))
            preds = (preds > 0.42).float()
        

        mask = preds.cpu().numpy().squeeze()
        

        upscale = A.Compose([
            A.Resize(originalShape[0], originalShape[1])
        ])
        
        fullsize = upscale(image=mask)['image']
        
        return fullsize
