import os
from PIL import Image

def create_transparent_png(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
        transparent_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        tumor_file_path = file_path.replace('.png', '_tumor.png')
        transparent_img.save(tumor_file_path)
        print(f"Created transparent image: {tumor_file_path}")

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png') and not filename.endswith('_tumor.png'):
            file_path = os.path.join(directory, filename)
            tumor_file_path = file_path.replace('.png', '_tumor.png')
            if not os.path.exists(tumor_file_path):
                create_transparent_png(file_path)

if __name__ == "__main__":
    directory_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Raw')
    process_directory(directory_path)
