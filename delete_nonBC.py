import os
import shutil
from natsort import natsorted

# Paths
input_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks"
output_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_ready"
os.makedirs(output_folder, exist_ok=True)


def find_missing_cases(folder):
    """Finds case numbers that are missing masks"""
    
    missing_cases = []
    
    # List images and masks as case IDs (case001, case002, etc.)
    image_files = natsorted([f.split(".png")[0] for f in os.listdir(folder) if f.endswith(".png") and "_tumor" not in f and "_other" not in f])
    mask_files = natsorted([f.split("_tumor.png")[0] for f in os.listdir(folder) if f.endswith(".png") and ("_tumor" in f or "_other" in f)])
    
    # Record case_id of cases without masks
    for case_id in image_files:
        if case_id not in mask_files:
            missing_cases.append(case_id)

    return missing_cases


def copy_images_with_masks(missing_cases):
    """Copies images with corresponding masks to a new folder and renames them so they are consecutive"""

    all_files = natsorted(os.listdir(input_folder))
    case_dict = {}

    for filename in all_files:
        case_id = filename.split(".png")[0].split("_tumor")[0].split("_other")[0]  # Extract case number
        
        if case_id in missing_cases:
            continue  # Skip missing cases
        
        if case_id not in case_dict:
            case_dict[case_id] = []
        
        case_dict[case_id].append(filename)

    # Copy files while renaming sequentially
    new_index = 1
    for case_id in natsorted(case_dict.keys()):  # Ensure order
        for filename in case_dict[case_id]:
            ext = filename.split(case_id)[-1]  # Keep original suffix (_tumor, _other, or nothing)
            new_filename = f"case{new_index:03d}{ext}"
            shutil.copy(os.path.join(input_folder, filename), os.path.join(output_folder, new_filename))

        new_index += 1  # Only increment after processing all files for a case


missing = find_missing_cases(input_folder)
copy_images_with_masks(missing)