import tensorflow as tf
import numpy as np
import tarfile
import cv2
from collections import defaultdict
import os
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, tar_fname, npz_fname, label_fname):
        self.tar_fname = tar_fname  # Now a single string
        self.npz_fname = npz_fname
        self.frame_ids, self.train_lbl, self.gaze_positions = self.read_label_file(label_fname)
        self.train_size = len(self.frame_ids)
        print(f"Dataset size: {self.train_size}")
        
        # Pre-load images and heatmaps
        self.images = self.preload_images()
        self.heatmaps = self.load_heatmaps()
        
    def read_label_file(self, label_fname):
        print("Reading in labels")
        frame_ids, lbls = [], []
        gaze_positions = defaultdict(list)
        
        with open(label_fname, 'r') as f:
            for line in f:
                if line.startswith("frame_id") or line.strip() == "":
                    continue
                dataline = line.strip().split(',')
                frame_id, lbl = dataline[0], dataline[5]
                try:
                    gaze_pos = [float(value) for value in dataline[6:] if value.strip()]
                    if gaze_pos:
                        gaze_positions[frame_id].extend(gaze_pos)
                except ValueError:
                    continue
                frame_ids.append(frame_id)
                lbls.append(int(lbl))
        
        return np.array(frame_ids), np.array(lbls, dtype=np.int32), gaze_positions

    def preprocess(self, image):
        # Resize the image and add channel dimension
        processed = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
        processed = processed.astype(np.float32) / 255.0
        return processed[:, :, np.newaxis]  # Add channel dimension
    
    def standardize(self, images):
        mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
        std = np.std(images, axis=(0, 1, 2), keepdims=True)
        return (images - mean) / (std + 1e-8)

    def preload_images(self):
        print("Preloading and standardizing images...")
        images = {}
        with tarfile.open(self.tar_fname, 'r') as tar:
            temp_images = []
            for member in tar.getmembers():
                if member.name.endswith('.png'):
                    f = tar.extractfile(member)
                    if f:
                        img_data = f.read()
                        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                        frame_id = os.path.splitext(os.path.basename(member.name))[0]
                        processed_img = self.preprocess(img)
                        temp_images.append(processed_img)
                        images[frame_id] = processed_img

            temp_images = np.array(temp_images)
            standardized_images = self.standardize(temp_images)
            
            for i, (frame_id, _) in enumerate(images.items()):
                images[frame_id] = standardized_images[i]
       
            
            # for i, (frame_id, _) in enumerate(images.items()):
            #     images[frame_id] = temp_images[i]

        print("Images preloaded and standardized")
        return images

    def load_heatmaps(self):
        print("Loading heatmaps...")
        data = np.load(self.npz_fname)
        heatmaps = data['heatmap']
        frame_ids = data['frame_ids']
        
        heatmap_dict = {}
        fallback_heatmap = None
        for i, frame_id in enumerate(frame_ids):
            heatmap = heatmaps[i]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
            heatmap_dict[frame_id] = heatmap[:, :, np.newaxis]  # Add channel dimension
            
            # Store the 4th frame's heatmap as fallback
            if i == 3:
                fallback_heatmap = heatmap_dict[frame_id]
        
        # Assign fallback heatmap to first three frames if missing
        for i in range(1, 4):
            frame_id = f"RZ_3560871_{i}"
            if frame_id not in heatmap_dict and fallback_heatmap is not None:
                heatmap_dict[frame_id] = fallback_heatmap
        
        print("Heatmaps loaded")
        return heatmap_dict

    def generator(self):
        for i, frame_id in enumerate(self.frame_ids):
            img = self.images.get(frame_id)
            
            if img is None:
                print(f"Skipping frame {frame_id} due to missing image")
                continue

            heatmap = self.heatmaps.get(frame_id)
            if heatmap is None:
                print(f"Skipping frame {frame_id} due to missing heatmap")
                continue

            label = self.train_lbl[i]
            
            yield (img, heatmap), label

    def get_dataset(self, batch_size, buffer_size):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(84, 84, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(84, 84, 1), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def get_steps_per_epoch(self, batch_size):
        return self.train_size // batch_size