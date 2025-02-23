from PIL import Image
import os
#print("Current working directory:", os.getcwd())


for folder in os.listdir("/Users/tscjake/Desktop/Fixed_Testing_DS"):
    if folder == ".DS_Store":
        continue
    elif folder == "benign":
        for image in os.listdir(f"/Users/tscjake/Desktop/Fixed_Testing_DS/{folder}"):
            print(f'found image: {image}')
            if image == ".DS_Store":
                print(f'skipped: {image}')
                continue
            oldname = f"/Users/tscjake/Desktop/Fixed_Testing_DS/{folder}/{image}"
            print(f'oldname: {oldname}')
            endlist = image.split(".")
            if 'png' in endlist:
                endlist.remove('png')
            newname = '.'.join(endlist)
            newname = f'/Users/tscjake/Desktop/Fixed_Testing_DS/{folder}/{newname}'
            print(f'newname is {newname}')
            os.rename(oldname,newname)
            print(f'photo renamed to {newname}')
    #image = Image.open(f"/Users/tscjake/Downloads/second_Dataset/normal/{i}").convert("RGBA")  
    #background = Image.new("RGB", image.size, (0, 0, 0))
    #background.paste(image, mask=image.split()[3])  
    #background = background.resize((256, 256))
    #background.save(f"/Users/tscjake/Desktop/normal/new_{i}.jpg", format="JPEG")
    #print(f"Saved as 'new_{i}.jpg'")

