import os
import numpy as np
from PIL import Image

# Combines all the masks for a given case
def combine_masks(case_id, folder_path):
    # List of all masks
    masks = []
    
    # Load primary tumor file
    primary_mask_path = os.path.join(folder_path, f"{case_id}_tumor.png")
    if os.path.exists(primary_mask_path):
        primary_mask = Image.open(primary_mask_path)  
        masks.append(np.array(primary_mask))
        
    # Load additional tumors
    for filename in os.listdir(folder_path):
        if filename.startswith(case_id) and filename.endswith(".png") and 'other' in filename:
            mask_path = os.path.join(folder_path, filename)
            mask = Image.open(mask_path)  
            masks.append(np.array(mask))
    
    # If no other tumor masks, return the original tumor mask
    if not masks:
        original_mask_path = os.path.join(folder_path, case_id + "_tumor.png")
        return Image.open(original_mask_path)  # Kept in RGB mode
    
    # Combine masks by taking the max pixel value (union of all tumor areas)
    # For each mask, if a pixel is white (255, 255, 255), we'll retain that
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)  # Combine the masks by taking max value for each pixel
    
    # Convert the combined mask back to an image
    combined_mask_image = Image.fromarray(combined_mask)
    return combined_mask_image

def save_combined_masks_and_cases(original_folder, target_folder):
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Process all cases and save combined masks and original images
    for filename in os.listdir(original_folder):
        if filename.endswith("_tumor.png"):  
            case_id = filename.split("_tumor.png")[0]  # Extract the case ID
            
            # Create combined mask for the case
            combined_mask = combine_masks(case_id, original_folder)
            
            # Save the combined mask with the original name (case_id_tumor.png) in the target folder
            combined_mask.save(os.path.join(target_folder, f"{case_id}_tumor.png"))
            
            # Copy the original ultrasound image of the case to the target folder
            original_image_path = os.path.join(original_folder, f"{case_id}.png")
            if os.path.exists(original_image_path):
                original_image = Image.open(original_image_path)
                original_image.save(os.path.join(target_folder, f"{case_id}.png"))

original_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks"  # Path to the original images and masks
target_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_ready"    # Path to the new folder where files will be saved
save_combined_masks_and_cases(original_folder, target_folder)

print("✅ Masks successfully combined")