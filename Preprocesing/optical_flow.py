import cv2
import numpy as np
import tarfile
import io
import re
import matplotlib.pyplot as plt

# Function to extract the numeric part of the filename for sorting
def extract_numeric_part(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else -1

def extract_non_numeric_part(filename):
    match = re.search(r'^(.*)_(\d+)\.png$', filename)
    return match.group(1) if match else filename

# Function to extract images from tar.bz2 file and get file names
def extract_images_from_tar(tar_path):
    file_images = []
    with tarfile.open(tar_path, 'r:bz2') as tar:
        for member in tar.getmembers():
            file = tar.extractfile(member)
            if file is not None:
                img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                file_images.append((member.name, img))
    # Sort images numerically by the number at the end of the filename
    file_images.sort(key=lambda x: (extract_non_numeric_part(x[0]), extract_numeric_part(x[0])))
    sorted_file_names, sorted_images = zip(*file_images)
    return list(sorted_images), list(sorted_file_names)

# Function to compute optical flow using Farnebäck method
def compute_optical_flow(images):
    optical_flow_images = []
    for i in range(len(images)):
        if i == 0:
            prev_img = images[i]
            next_img = images[i]  # Use the same image for the first case
        else:
            prev_img = images[i - 1]
            next_img = images[i]

        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None,
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # Convert flow to RGB for visualization purposes
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR))
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        optical_flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        optical_flow_images.append(optical_flow_img)
    return optical_flow_images

# Function to save optical flow images into a tar.bz2 file with the same names as the original images
def save_optical_flow_to_tar(optical_flow_images, file_names, output_tar_path):
    with tarfile.open(output_tar_path, 'w:bz2') as tar:
        for img, name in zip(optical_flow_images, file_names):
            img_encoded = cv2.imencode('.png', img)[1].tobytes()
            img_info = tarfile.TarInfo(name)
            img_info.size = len(img_encoded)
            tar.addfile(img_info, io.BytesIO(img_encoded))

# Function to plot images
def plot_images(original_images, optical_flow_images):
    plt.figure(figsize=(12, 8))
    for i in range(min(3, len(original_images))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title(f'Original Image {i+1}')
        plt.axis('off')
        
        plt.subplot(2, 3, i + 4)
        plt.imshow(optical_flow_images[i])
        plt.title(f'Optical Flow Image {i+1}')
        plt.axis('off')
    plt.show()

# Main function
def main():
    # tar_path = './ms_pacman/combined_data.tar.bz2'
    # output_tar_path = './ms_pacman/combined_data_opt.tar.bz2'
    tar_path = './ms_pacman/test.tar.bz2'
    output_tar_path = './ms_pacman/test_opt.tar.bz2'
    
    # Extract images and file names
    images, file_names = extract_images_from_tar(tar_path)
    
    # Compute optical flow
    optical_flow_images = compute_optical_flow(images)
    
    # Save optical flow images to tar.bz2 with the original names
    save_optical_flow_to_tar(optical_flow_images, file_names, output_tar_path)
    
    # Plot the first 3 images and their corresponding optical flow
    plot_images(images[:3], optical_flow_images[:3])

if __name__ == "__main__":
    main()
