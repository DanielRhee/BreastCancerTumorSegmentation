import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("mps found")
else:
    print("cpu time :(")
    DEVICE = torch.device("cpu")

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

def create_overlay(original_image, mask, alpha=0.5, color=[1, 0, 0]):
    if original_image.max() > 1:
        original_image = original_image / 255.0
    
    colored_mask = np.zeros_like(original_image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]
    
    overlay = original_image.copy()
    mask_3d = np.stack([mask] * 3, axis=-1)
    overlay[mask_3d > 0] = (1 - alpha) * original_image[mask_3d > 0] + alpha * colored_mask[mask_3d > 0]
    
    return overlay

def predict_mask(model, image_raw, device="cpu"):
    model.eval()
    originalShape = image_raw.shape

    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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
        preds = torch.sigmoid(model(image_tensor))
        preds = (preds > 0.42).float()
    

    mask = preds.cpu().numpy().squeeze()
    

    upscale = A.Compose([
        A.Resize(originalShape[0], originalShape[1])
    ])
    
    fullsize = upscale(image=mask)['image']
    
    return fullsize

def loadModel():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("cache.pth.tar", map_location=DEVICE)["state_dict"])
    return model

def predict_and_visualize(image_path, model, save_path=None):

    img = np.array(Image.open(image_path).convert("RGB"))
    

    mask = predict_mask(model, img, device=DEVICE)

    print(mask)
    

    overlay = create_overlay(img, mask)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(img)
    ax1.set_title('Original Image')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Predicted Mask')

    ax3.imshow(overlay)
    ax3.set_title('Overlay')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    model = loadModel()
    
    image_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Raw') + "/case014.png"
    save_path = 'pred.png'
    
    predict_and_visualize(image_path, model, save_path)
