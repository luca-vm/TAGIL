import os
import matplotlib.pyplot as plt
import numpy as np

def match_and_rename_predictions(npz_file, input_dir, output_dir):
    # Load predictions, durations, and frame IDs
    data = np.load(npz_file)
    predictions = data['heatmap']
    durations = data['durations']
    frame_ids = data['frame_ids']

    # Define duration bins
    bins = [60]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, frame_id in enumerate(frame_ids):
        # Generate the corresponding input image name
        
        # Get the duration prediction
        duration_pred = durations[i]
        duration_bin = np.argmax(duration_pred)
        
        # Map duration bin to actual duration value
        if duration_bin == 0:
            duration_value = "easy"
        elif duration_bin == len(bins):
            duration_value = "hard"
        else:
            duration_value = f"{bins[duration_bin-1]}-{bins[duration_bin]}"
        
        # Generate the new name for the prediction image
        prediction_image_name = f"gaze_{duration_value}_{frame_id}.png"
        
        
        # Plot gaze heatmap
        heatmap = predictions[i]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
        plt.imsave(os.path.join(output_dir, prediction_image_name), heatmap, cmap='inferno')
      
        
        # print(f"Saved prediction for input image {input_image_name} as {prediction_image_name}")

# Usage
npz_file = "./ms_pacman/gaze_smote_test.npz"
input_dir = "./ms_pacman/test"  # Directory containing original input images
output_dir = "./ms_pacman/predictions_smote"  # Directory to save renamed prediction images

match_and_rename_predictions(npz_file, input_dir, output_dir)