import cv2
import numpy as np
import tarfile
import io
import os
import re
import matplotlib.pyplot as plt


# Function to extract the numeric part of the filename for sorting
def extract_numeric_part(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else -1

def extract_non_numeric_part(filename):
    match = re.search(r'^(.*)_(\d+)\.png$', filename)
    return match.group(1) if match else filename

# Function to check if the file is an image (by file extension)
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

# Function to extract images from tar.bz2 file and get file names, including subdirectories
def extract_images_from_tar(tar_path):
    file_images = []
    with tarfile.open(tar_path, 'r:bz2') as tar:
        for member in tar.getmembers():
            # Check if it's a file (not a directory) and it's an image file
            if member.isfile() and is_image_file(member.name):
                file = tar.extractfile(member)
                if file is not None:
                    img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    file_images.append((member.name, img))
                    
    # Sort images numerically by the number at the end of the filename
    file_images.sort(key=lambda x: (extract_non_numeric_part(x[0]), extract_numeric_part(x[0])))
    sorted_file_names, sorted_images = zip(*file_images)
    return list(sorted_images), list(sorted_file_names)


# Function to compute saliency maps using Farnebäck method
def compute_saliency(images):
    sal_images = []
    for i in range(len(images)):   
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        (success, saliencyMap) = saliency.computeSaliency(images[i])

        saliencyMap = (saliencyMap * 255).astype("uint8")
        sal_images.append(saliencyMap)
    return sal_images

# Function to save saliency images into a tar.bz2 file with the same names as the original images
def save_saliency_to_tar(sal_images, file_names, output_tar_path):
    with tarfile.open(output_tar_path, 'w:bz2') as tar:
        for img, name in zip(sal_images, file_names):
            img_encoded = cv2.imencode('.png', img)[1].tobytes()
            img_info = tarfile.TarInfo(name=name.replace('.png', '.png'))
            img_info.size = len(img_encoded)
            tar.addfile(img_info, io.BytesIO(img_encoded))

# Function to plot images
def plot_images(original_images, sal_images):
    plt.figure(figsize=(12, 8))
    for i in range(min(3, len(original_images))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title(f'Saliency Image {i+1}')
        plt.axis('off')
        
        plt.subplot(2, 3, i + 4)
        plt.imshow(sal_images[i])
        plt.title(f'Saliency Flow Image {i+1}')
        plt.axis('off')
    plt.show()

# Main function
def main():
    # tar_path = './ms_pacman/val.tar.bz2'
    # output_tar_path = './ms_pacman/test_sal.tar.bz2'
    tar_path = '/datasets/lvmayer/berzerk/val/val.tar.bz2'
    output_tar_path = '/datasets/lvmayer/berzerk/val/val_sal.tar.bz2'
    
    # Extract images and file names (including subdirectories)
    images, file_names = extract_images_from_tar(tar_path)
    
    # Compute saliency maps
    sal_images = compute_saliency(images)
    
    # Save saliency images to tar.bz2 with the original names
    save_saliency_to_tar(sal_images, file_names, output_tar_path)
    
    # Plot the first 3 images and their corresponding saliency maps
    # plot_images(images[:3], sal_images[:3])

if __name__ == "__main__":
    main()
