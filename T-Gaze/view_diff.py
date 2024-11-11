import numpy as np
from collections import defaultdict

def read_label_file(label_fname):
    print("Reading in labels...")
    frame_ids, lbls, durations = [], [], []
    
    with open(label_fname, 'r') as f:
        for line in f:
            if line.startswith("frame_id") or line.strip() == "":
                continue
            dataline = line.strip().split(',')
            frame_id, duration, lbl = dataline[0], dataline[3], dataline[5]
            try:
                frame_ids.append(frame_id)
                lbls.append(int(lbl))
                durations.append(int(duration))
            except ValueError:
                continue
    
    return np.array(frame_ids), np.array(lbls, dtype=np.int32), np.array(durations)

def get_frame_ids_in_duration_bin(label_fname, min_duration=60):
    frame_ids, _, durations = read_label_file(label_fname)
    
    # Get frame_ids and durations for those in the 60+ bin
    qualifying_frame_ids = [(frame_id, duration) for frame_id, duration in zip(frame_ids, durations) if duration >= min_duration]
    
    return qualifying_frame_ids

# Example usage
if __name__ == "__main__":
    # label_fname = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35.txt'  # Replace with your label file path
    label_fname = './ms_pacman/test.txt'  # Replace with your label file path
    qualifying_frames = get_frame_ids_in_duration_bin(label_fname)
    print(len(qualifying_frames))
    print("Frame IDs and Durations for 60+ seconds:")
    for frame_id, duration in qualifying_frames:
        print(f"Frame ID: {frame_id}, Duration: {duration} seconds")