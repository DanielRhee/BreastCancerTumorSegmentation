import predict 
from PIL import Image

obj = predict.predictor()

image = int(input())

print(obj.predict(Image.open("Data/Raw/Case001.png").convert("RGB")))
