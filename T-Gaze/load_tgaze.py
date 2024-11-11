import tensorflow as tf
import numpy as np
import tarfile
import cv2
from collections import deque, defaultdict
import os
import multiprocessing

class Dataset:
    def __init__(self, tar_fname, opt_fname, sal_fname, label_fname):
        self.tar_fname = tar_fname
        self.opt_fname = opt_fname
        self.sal_fname = sal_fname
        self.frame_ids, self.train_lbl, self.gaze_positions, self.duration_bins = self.read_label_file(label_fname)
        self.train_size = len(self.frame_ids)
        print(f"Dataset size: {self.train_size}")
        
        # Pre-load images
        self.images = self.preload_images()

    def bin_durations(self, durations):
        bins = [
            (0, 60),    # Duration < 60
            (60, float('inf'))  # Duration >= 60
        ]
        binned_durations = np.zeros((len(durations), len(bins)), dtype=np.float32)
        for i, duration in enumerate(durations):
            for j, (low, high) in enumerate(bins):
                if low <= duration < high:
                    binned_durations[i, j] = 1
                    break
        return binned_durations

    
    def read_label_file(self, label_fname):
        print("Reading in labels")
        frame_ids, lbls, durations = [], [], []
        gaze_positions = defaultdict(list)
        
        with open(label_fname, 'r') as f:
            for line in f:
                if line.startswith("frame_id") or line.strip() == "":
                    continue
                dataline = line.strip().split(',')
                frame_id, duration, lbl = dataline[0], dataline[3], dataline[5]
                try:
                    gaze_pos = [float(value) for value in dataline[6:] if value.strip()]
                    if gaze_pos:
                        gaze_positions[frame_id].extend(gaze_pos)
                except ValueError:
                    continue
                frame_ids.append(frame_id)
                lbls.append(int(lbl))
                durations.append(int(duration))
        
        duration_bins = self.bin_durations(durations)
        
        # Print the length of each bin
        bin_lengths = np.sum(duration_bins, axis=0)
        print(f"Number of durations in bin 1 (< 60ms): {bin_lengths[0]}")
        print(f"Number of durations in bin 2 (>= 60ms): {bin_lengths[1]}")
        
        return np.array(frame_ids), np.array(lbls, dtype=np.int32), gaze_positions, duration_bins

    def preprocess(self, image):
        return cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    
    def standardize(self, images):
        """Standardize images to have zero mean and unit variance."""
        mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
        std = np.std(images, axis=(0, 1, 2), keepdims=True)
        return (images - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

    def read_image_from_tar(self, tar_fname, frame_id):
        with tarfile.open(tar_fname, 'r') as tar:
            img_name = f'{frame_id}.png'
            for member in tar.getmembers():
                if member.name.endswith(img_name):
                    f = tar.extractfile(member)
                    if f:
                        img_data = f.read()
                        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                        return self.preprocess(img)
        print(f"Image {img_name} not found in {tar_fname}")
        return None

    def preload_images(self):
        print("Preloading and standardizing images...")
        images = {}
        for tar_file in [self.tar_fname, self.opt_fname, self.sal_fname]:
            images[tar_file] = {}
            with tarfile.open(tar_file, 'r') as tar:
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
                            images[tar_file][frame_id] = processed_img

                # Standardize all images for this tar file
                temp_images = np.array(temp_images)
                standardized_images = self.standardize(temp_images)
                
                # Update the images dictionary with standardized images
                for i, (frame_id, _) in enumerate(images[tar_file].items()):
                    images[tar_file][frame_id] = standardized_images[i]

        print("Images preloaded and standardized")
        return images

    def generate_gaze_map(self, frame_id):
        gaze_map = np.zeros((84, 84, 1), dtype=np.float32)
        positions = self.gaze_positions.get(frame_id, [])

        for x, y in zip(positions[::2], positions[1::2]):
            x, y = int(x * 84 / 160), int(y * 84 / 210)
            if 0 <= x < 84 and 0 <= y < 84:
                # Create a temporary map with a bright center at the gaze position
                temp_map = np.zeros((84, 84), dtype=np.float32)
                cv2.circle(temp_map, (x, y), 1, 1, -1)  # A small bright circle in the center

                # Apply Gaussian blur to create a smooth fading effect
                temp_map = cv2.GaussianBlur(temp_map, (17, 17), 0)

                # Add the blurred map to the gaze map
                gaze_map[:, :, 0] += temp_map

        # Normalize the gaze map to keep values between 0 and 1
        gaze_map = np.clip(gaze_map, 0, 1)

        return gaze_map

    def create_stacked_obs(self, imgs):
        stack = list(imgs)
        if len(stack) < 4:
            # Duplicate the last frame (most recent) to fill the stack
            last_frame = stack[-1]
            while len(stack) < 4:
                stack.insert(0, last_frame)
        return np.stack(stack, axis=-1)

    def generator(self):
        imgs_deque = deque(maxlen=4)
        opt_imgs_deque = deque(maxlen=4)
        sal_imgs_deque = deque(maxlen=4)

        for i, frame_id in enumerate(self.frame_ids):
            img = self.images[self.tar_fname].get(frame_id)
            opt_img = self.images[self.opt_fname].get(frame_id)
            sal_img = self.images[self.sal_fname].get(frame_id)
            
            if img is None or opt_img is None or sal_img is None:
                print(f"Skipping frame {frame_id} due to missing image")
                continue

            imgs_deque.append(img)
            opt_imgs_deque.append(opt_img)
            sal_imgs_deque.append(sal_img)

            stacked_img = self.create_stacked_obs(imgs_deque)
            stacked_opt = self.create_stacked_obs(opt_imgs_deque)
            stacked_sal = self.create_stacked_obs(sal_imgs_deque)
            
            gaze_map = self.generate_gaze_map(frame_id)
            duration_bin = self.duration_bins[i]
            
            combined_label = tf.concat([tf.reshape(gaze_map, [-1]), duration_bin], axis=0)
            
            yield (
                (
                    tf.convert_to_tensor(stacked_img, dtype=tf.float32),
                    tf.convert_to_tensor(stacked_opt, dtype=tf.float32),
                    tf.convert_to_tensor(stacked_sal, dtype=tf.float32)
                ),
                tf.convert_to_tensor(combined_label, dtype=tf.float32)
            )


    def get_dataset(self, batch_size, buffer_size):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(84, 84, 4), dtype=tf.float32),
                    tf.TensorSpec(shape=(84, 84, 4), dtype=tf.float32),
                    tf.TensorSpec(shape=(84, 84, 4), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(84*84*1 + 2,), dtype=tf.float32)
            )
        )
        
        dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        return dataset
    


    def get_steps_per_epoch(self, batch_size):
        return self.train_size // batch_size
    