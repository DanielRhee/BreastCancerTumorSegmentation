import os
import numpy as np
from PIL import Image
import shutil
from natsort import natsorted


# Paths
input_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_ready"
output_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc"
os.makedirs(output_folder, exist_ok=True)


def combine_masks(folder):
    """Combines all mask variations into a single tumor mask per case, preserving original background format."""

    all_files = natsorted(os.listdir(folder))
    mask_dict = {}

    # Sort and group masks by case ID
    for filename in all_files:
        if filename.endswith(".png") and ("_tumor" in filename or "_other" in filename):
            case_id = filename.split("_tumor")[0].split("_other")[0]
            if case_id not in mask_dict:
                mask_dict[case_id] = []
            mask_dict[case_id].append(filename)

    for case_id in natsorted(mask_dict.keys()):  # Process in order
        mask_files = natsorted(mask_dict[case_id])  # Ensure masks are ordered

        if len(mask_files) > 1:
            masks = [np.array(Image.open(os.path.join(folder, mf)).convert("L")) for mf in mask_files]

            # Combine masks by taking the max pixel value (union of all tumor areas)
            combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
            for mask in masks:
                combined_mask = np.maximum(combined_mask, mask)

            # Ensure background stays consistent with the original format
            combined_mask = np.where(combined_mask > 0, 255, 150).astype(np.uint8)

            # Save the final combined mask as caseXXX_tumor.png
            combined_mask_image = Image.fromarray(combined_mask, mode="L")
            final_mask_path = os.path.join(folder, f"{case_id}_tumor.png")
            combined_mask_image.save(final_mask_path)

            # Delete extra masks (keep only the new caseXXX_tumor.png)
            for mask_file in mask_files:
                if mask_file != f"{case_id}_tumor.png":
                    os.remove(os.path.join(folder, mask_file))


def copy_and_process_files():
    """Copies all files to the new folder and processes masks"""
    
    all_files = natsorted(os.listdir(input_folder))

    for filename in all_files:
        shutil.copy(os.path.join(input_folder, filename), os.path.join(output_folder, filename))

    combine_masks(output_folder)


copy_and_process_files()