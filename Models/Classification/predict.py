import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import ClassifyCNN
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

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# Load the model
def loadModel():
    model = ClassifyCNN(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load("classify.pth.tar", map_location=DEVICE)["state_dict"])
    model.eval()
    return model

# Preprocess image with albumentations
def preprocess_image(image):
    transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Update this if needed
        ToTensorV2()
    ])
    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0).to(DEVICE)

# Predict the class for an image
def predict(model, image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()  # Get predicted class
        probabilities = torch.softmax(outputs, dim=1)  # Get prediction probabilities
    return predicted_class, probabilities

# Main program to load model and predict for each image
if __name__ == "__main__":
    model = loadModel()
    
    for i in range(1, 256):
        path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Raw', f"case{str(i).zfill(3)}.png")
        
        # Load the image and convert it to RGB
        image = np.array(Image.open(path).convert("RGB"))
        
        # Make prediction
        prediction, probabilities = predict(model, image)
        
        # Print predicted class and its probability
        print(f"Predicted Class: {prediction} (Probabilities: {probabilities.squeeze().cpu().numpy()})")
    
