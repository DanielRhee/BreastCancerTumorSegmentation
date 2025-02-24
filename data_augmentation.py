import os
import shutil


# Dataset folders after augmentation
folders = [
    "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc",           # Folder 1 (0-degree rotation)
    "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc_90",        # Folder 2 (90-degree rotation)
    "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc_180",       # Folder 3 (180-degree rotation)
    "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_readyc_270"        # Folder 4 (270-degree rotation)
]

output_folder = "/Users/macuser/Documents/GitHub/BreastCancerTumorSegmentation/images/BC_imgsmasks_final" # Final Folder
os.makedirs(output_folder, exist_ok=True)


def count_cases(original_folder):
    """Determines how many cases are in each folder"""
    
    cases = 0
    
    # Count number of files in folder
    for path in os.scandir(original_folder):
        if path.is_file():
            cases += 1
            
    if cases % 2 != 0:
        print("There are an an odd number of files in the folder. Something went wrong while combining masks or filtering images.")
        return None
    return (int)(cases/2)


def copy_original_folder(original_folder, output_dir):
    """Copies the contents of original folder into final folder, maintaining order"""
    
    original_folder_files = os.listdir(original_folder)
    for filename in original_folder_files:
        source_path = os.path.join(original_folder, filename)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy(source_path, dest_path)
        
        
def update_case_ids(folder, offset):
    """Updates the case IDs for a given folder by adding an offset to the ID number"""
    
    folder_files = sorted(os.listdir(folder))
    
    for filename in folder_files:
        case_id = filename.split("_")[0].split(".")[0]
        case_number = int(case_id.replace("case", "")) # Typecast
        new_case_number = case_number + offset
        
        new_case_id = f"case{new_case_number:03d}"  # Format back to caseXXX
        new_filename = filename.replace(case_id, new_case_id, 1)
        
        # Copy file with new case ID to final folder
        source_path = os.path.join(folder, filename)
        destination_path = os.path.join(output_folder, new_filename)
        shutil.copy(source_path, destination_path)
    

num_cases = count_cases(folders[0])
if num_cases is not None:
    copy_original_folder(folders[0], output_folder)

    case_offsets = [num_cases, num_cases*2, num_cases*3]
    for i, folder in enumerate(folders[1:],1):            # Skip folder 1
        update_case_ids(folder, case_offsets[i-1])



    