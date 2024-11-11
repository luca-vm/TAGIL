import numpy as np
from collections import deque
import tensorflow as tf
from load_tgaze import Dataset
import matplotlib.pyplot as plt

class TemporalSMOTEDataset(Dataset):
    def __init__(self, tar_fname, opt_fname, sal_fname, label_fname, minority_increase=0.2):
        super().__init__(tar_fname, opt_fname, sal_fname, label_fname)
        
        # Generate gaze maps for all frames
        self.gaze_maps = {frame_id: self.generate_gaze_map(frame_id) for frame_id in self.frame_ids}
        
        # Identify minority class samples (duration > 60ms)
        self.minority_indices = np.where(self.duration_bins[:, 1] == 1)[0]
        
        print(f"Number of minority samples before SMOTE: {len(self.minority_indices)}")
        
        num_synthetic = int(len(self.minority_indices) * minority_increase)
        print(f"Number of synthetic samples to generate: {num_synthetic}")
        
        self.synthetic_samples = self.generate_synthetic_samples(num_synthetic)
        
        print(f"Number of synthetic samples actually generated: {len(self.synthetic_samples)}")
        print(f"Original dataset size: {len(self.frame_ids)}")
        print(f"Dataset size after Temporal SMOTE: {len(self.frame_ids) + len(self.synthetic_samples)}")

    def generate_synthetic_samples(self, num_synthetic):
        synthetic_samples = []
        minority_sample_count = len(self.minority_indices)
        
        for _ in range(num_synthetic):
            idx = self.minority_indices[np.random.randint(minority_sample_count)]
            
            if idx > 0 and idx < len(self.frame_ids) - 1:
                prev_frame_id = self.frame_ids[idx-1]
                curr_frame_id = self.frame_ids[idx]
                next_frame_id = self.frame_ids[idx+1]
                
                # Extract gameplay ID and frame numbers
                prev_gameplay_id, prev_frame_num = self.split_frame_id(prev_frame_id)
                curr_gameplay_id, curr_frame_num = self.split_frame_id(curr_frame_id)
                next_gameplay_id, next_frame_num = self.split_frame_id(next_frame_id)
                
                # Only interpolate if the gameplay IDs match
                if prev_gameplay_id == curr_gameplay_id == next_gameplay_id:
                    # Randomly choose to create a sample before or after the minority sample
                    if np.random.random() < 0.5:
                        synth_frame_num = (prev_frame_num + curr_frame_num) // 2
                        synth_frame_id = f"{curr_gameplay_id}_{synth_frame_num:04d}"
                        prev_id, next_id = prev_frame_id, curr_frame_id
                    else:
                        synth_frame_num = (curr_frame_num + next_frame_num) // 2
                        synth_frame_id = f"{curr_gameplay_id}_{synth_frame_num:04d}"
                        prev_id, next_id = curr_frame_id, next_frame_id
                    
                    synthetic_samples.append({
                        'synth_frame_id': synth_frame_id,
                        'prev_frame_id': prev_id,
                        'next_frame_id': next_id,
                        'duration_bin': self.duration_bins[idx]
                    })
        
        return synthetic_samples

    @staticmethod
    def split_frame_id(frame_id):
        parts = frame_id.rsplit('_', 1)
        return parts[0], int(parts[1])

    def interpolate_images(self, prev_img, next_img):
        return (prev_img + next_img) / 2

    def interpolate_gaze_maps(self, prev_gaze_map, next_gaze_map):
        return (prev_gaze_map + next_gaze_map) / 2

    def generator(self):
        imgs_deque = deque(maxlen=4)
        opt_imgs_deque = deque(maxlen=4)
        sal_imgs_deque = deque(maxlen=4)

        # Yield original samples
        for i, frame_id in enumerate(self.frame_ids):
            yield self.generate_sample(frame_id, i, imgs_deque, opt_imgs_deque, sal_imgs_deque)

        # Yield synthetic samples
        for sample in self.synthetic_samples:
            prev_frame_id = sample['prev_frame_id']
            next_frame_id = sample['next_frame_id']
            synth_frame_id = sample['synth_frame_id']

            # Interpolate images
            img = self.interpolate_images(self.images[self.tar_fname][prev_frame_id], self.images[self.tar_fname][next_frame_id])
            opt_img = self.interpolate_images(self.images[self.opt_fname][prev_frame_id], self.images[self.opt_fname][next_frame_id])
            sal_img = self.interpolate_images(self.images[self.sal_fname][prev_frame_id], self.images[self.sal_fname][next_frame_id])

            # Interpolate gaze map
            gaze_map = self.interpolate_gaze_maps(self.gaze_maps[prev_frame_id], self.gaze_maps[next_frame_id])

            imgs_deque.append(img)
            opt_imgs_deque.append(opt_img)
            sal_imgs_deque.append(sal_img)

            stacked_img = self.create_stacked_obs(imgs_deque)
            stacked_opt = self.create_stacked_obs(opt_imgs_deque)
            stacked_sal = self.create_stacked_obs(sal_imgs_deque)

            duration_bin = sample['duration_bin']

            combined_label = tf.concat([tf.reshape(gaze_map, [-1]), duration_bin], axis=0)

            yield (
                (
                    tf.convert_to_tensor(stacked_img, dtype=tf.float32),
                    tf.convert_to_tensor(stacked_opt, dtype=tf.float32),
                    tf.convert_to_tensor(stacked_sal, dtype=tf.float32)
                ),
                tf.convert_to_tensor(combined_label, dtype=tf.float32)
            )

    def generate_sample(self, frame_id, index, imgs_deque, opt_imgs_deque, sal_imgs_deque):
        img = self.images[self.tar_fname][frame_id]
        opt_img = self.images[self.opt_fname][frame_id]
        sal_img = self.images[self.sal_fname][frame_id]

        imgs_deque.append(img)
        opt_imgs_deque.append(opt_img)
        sal_imgs_deque.append(sal_img)

        stacked_img = self.create_stacked_obs(imgs_deque)
        stacked_opt = self.create_stacked_obs(opt_imgs_deque)
        stacked_sal = self.create_stacked_obs(sal_imgs_deque)

        gaze_map = self.gaze_maps[frame_id]
        duration_bin = self.duration_bins[index]

        combined_label = tf.concat([tf.reshape(gaze_map, [-1]), duration_bin], axis=0)

        return (
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
        return (len(self.frame_ids) + len(self.synthetic_samples)) // batch_size
    
    def visualize_synthetic_sample(self, sample_index=0):
        if not self.synthetic_samples:
            print("No synthetic samples available.")
            return

        sample = self.synthetic_samples[sample_index]
        prev_frame_id = sample['prev_frame_id']
        next_frame_id = sample['next_frame_id']

        # Prepare data for visualization
        prev_img = self.images[self.tar_fname][prev_frame_id]
        next_img = self.images[self.tar_fname][next_frame_id]
        synth_img = self.interpolate_images(prev_img, next_img)

        prev_opt = self.images[self.opt_fname][prev_frame_id]
        next_opt = self.images[self.opt_fname][next_frame_id]
        synth_opt = self.interpolate_images(prev_opt, next_opt)

        prev_sal = self.images[self.sal_fname][prev_frame_id]
        next_sal = self.images[self.sal_fname][next_frame_id]
        synth_sal = self.interpolate_images(prev_sal, next_sal)

        prev_gaze = self.gaze_maps[prev_frame_id]
        next_gaze = self.gaze_maps[next_frame_id]
        synth_gaze = self.interpolate_gaze_maps(prev_gaze, next_gaze)

        # Create the plot
        fig, axs = plt.subplots(5, 3, figsize=(15, 25))
        fig.suptitle(f"Synthetic Sample Visualization (Index: {sample_index})", fontsize=16)

        # Helper function to plot images
        def plot_image(ax, img, title):
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        # Plot images
        plot_image(axs[0, 0], prev_img, f"Previous Frame\n{prev_frame_id}")
        plot_image(axs[0, 1], synth_img, "Synthetic Frame")
        plot_image(axs[0, 2], next_img, f"Next Frame\n{next_frame_id}")

        # Plot optical flow
        plot_image(axs[1, 0], prev_opt, "Previous Optical Flow")
        plot_image(axs[1, 1], synth_opt, "Synthetic Optical Flow")
        plot_image(axs[1, 2], next_opt, "Next Optical Flow")

        # Plot saliency maps
        plot_image(axs[2, 0], prev_sal, "Previous Saliency")
        plot_image(axs[2, 1], synth_sal, "Synthetic Saliency")
        plot_image(axs[2, 2], next_sal, "Next Saliency")

        # Plot gaze maps
        plot_image(axs[3, 0], prev_gaze, "Previous Gaze Map")
        plot_image(axs[3, 1], synth_gaze, "Synthetic Gaze Map")
        plot_image(axs[3, 2], next_gaze, "Next Gaze Map")

        # Plot durations
        prev_duration = self.duration_bins[np.where(self.frame_ids == prev_frame_id)[0][0]]
        next_duration = self.duration_bins[np.where(self.frame_ids == next_frame_id)[0][0]]
        synth_duration = sample['duration_bin']

        axs[4, 0].bar(['<60ms', '>=60ms'], prev_duration)
        axs[4, 0].set_title("Previous Duration")
        axs[4, 1].bar(['<60ms', '>=60ms'], synth_duration)
        axs[4, 1].set_title("Synthetic Duration")
        axs[4, 2].bar(['<60ms', '>=60ms'], next_duration)
        axs[4, 2].set_title("Next Duration")

        plt.tight_layout()
        plt.show()