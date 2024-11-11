# import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

def match_and_rename_predictions(npz_file, output_dir):
    # Load predictions and frame IDs
    data = np.load(npz_file)
    predictions = data['heatmap']
    frame_ids = data['frame_ids']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, frame_id in enumerate(frame_ids):
        
        # Generate the new name for the prediction image
        prediction_image_name = f"gaze_{frame_id}.png"
        
        # Save the prediction as an image with the new name
        heatmap = predictions[i]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
        plt.imsave(os.path.join(output_dir, prediction_image_name), heatmap, cmap='inferno')
        
        # print(f"Saved prediction for input image {input_image_name} as {prediction_image_name}")

# Usage
npz_file = "./ms_pacman/gaze_test.npz"
output_dir = "./ms_pacman/predictions"  # Directory to save renamed prediction images

match_and_rename_predictions(npz_file, output_dir)