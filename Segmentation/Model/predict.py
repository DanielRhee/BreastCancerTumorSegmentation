import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
import numpy as np
from PIL import Image

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("mps found")
else:
    print ("cpu time :(")
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256 


def predict_mask(model, image_raw, device="cpu"):
    model.eval()

    #image_tensor = torch.as_tensor(np.array(image_raw).astype('float32'))
    originalShape = image_raw.shape

    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]) 

    image_tensor = transform(image=image_raw)["image"]


    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    #print(f"Final image tensor shape: {image_tensor.shape}")
    
    with torch.no_grad():
        preds = torch.sigmoid(model(image_tensor))
        preds = (preds > 0.5).float()
    
    # Convert to numpy array and remove batch dimension
    mask = preds.cpu().numpy().squeeze()
        
    upscale = A.Compose([
        A.Resize(originalShape[0], originalShape[1])
    ])
    
    fullsize =  upscale(image=mask)['image']

    return fullsize



def loadModel():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("cache.pth.tar", weights_only=True)["state_dict"])
    
    return model


if __name__ == "__main__":
    path = ''
    model = loadModel()
    img = np.array(Image.open('path').convert("RGB"))
    mask = predict_mask(model, img, device=torch.device("mps"))
    from matplotlib import pyplot as plt
    plt.imshow(mask)
    plt.show()
