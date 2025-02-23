import os
from PIL import Image


def rotate_folder(input_folder, output_folder, angle):
    """Saves rotated files in a folder to a new folder"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with Image.open(file_path) as img: # Open the image
                    rotated_img = img.rotate(angle, expand=True) # Rotate the image
                    new_file_path = os.path.join(output_folder, filename)
                    rotated_img.save(new_file_path)
                    
            except Exception as e:
                print(f"Error rotating {filename}: {e}")
            
            
input_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc"
output_folder_90 = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc_90"
output_folder_180 = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc_180"
output_folder_270 = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc_270"
rotate_folder(input_folder, output_folder_90, 90)
rotate_folder(input_folder, output_folder_180, 180)
rotate_folder(input_folder, output_folder_270, 270)
