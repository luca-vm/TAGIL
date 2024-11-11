import os
import time
import shutil
import data_reader
import tarfile
import pygame
import numpy as np

# Control variables
SHOW_BASELINE = True
SHOW_DURATIONS = True

class DrawgcWrapper:
    def __init__(self):
        self.cursor = pygame.image.load('target.png')
        self.cursor_size = (self.cursor.get_width(), self.cursor.get_height())

    def draw_gc(self, screen, gaze_position):
        region_top_left = (gaze_position[0] - self.cursor_size[0] // 2, gaze_position[1] - self.cursor_size[1] // 2)
        screen.blit(self.cursor, region_top_left)

class DrawingStatus:
    def __init__(self):
        self.cur_frame_index = 0
        self.total_frame = 0
        self.target_fps = 60
        self.pause = False
        self.terminated = False

ds = DrawingStatus()

def preprocess_and_sanity_check(png_files, frameid_list):
    has_warning = False

    # remove unrelated files
    for fname in png_files:
        if not fname.endswith(".png"):
            print("Warning: %s is not a PNG file. Deleting it from the frames list" % fname)
            has_warning = True
            png_files.remove(fname)

    # check if each frame has its corresponding png file
    based_dir = png_files[0].split('/')[0]
    set_png_files = set(png_files)
    for frameid in frameid_list:
        png_fname = based_dir + '/' + frameid + '.png'
        if png_fname not in set_png_files:
            print("Warning: no corresponding png file for frame id %s." % frameid)
            has_warning = True

    if has_warning:
        print("There are warnings. Sleeping for 2 sec...")
        time.sleep(2)


def frameid_sort_key(frameid):
    return int(frameid.split('_')[2])

def check_gaze_range(pos_x, pos_y, w, h):
    return 0 <= pos_x <= w and 0 <= pos_y <= h

def event_handler_func():
    global ds
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                ds.cur_frame_index = max(0, ds.cur_frame_index - 5 * ds.target_fps)
            elif event.key == pygame.K_DOWN:
                ds.cur_frame_index = min(ds.total_frame - 1, ds.cur_frame_index + 5 * ds.target_fps)
            elif event.key == pygame.K_LEFT:
                ds.cur_frame_index = max(0, ds.cur_frame_index - 1)
            elif event.key == pygame.K_RIGHT:
                ds.cur_frame_index = min(ds.total_frame - 1, ds.cur_frame_index + 1)
            elif event.key == pygame.K_SPACE:
                ds.pause = not ds.pause
            elif event.key == pygame.K_ESCAPE:
                ds.terminated = True

def create_heatmap_surface(heatmap, size, color):
    heatmap = pygame.transform.scale(heatmap, size)
    heatmap_surface = pygame.Surface(size, pygame.SRCALPHA)
    for x in range(size[0]):
        for y in range(size[1]):
            value = heatmap.get_at((x, y))[0]  # Get red channel as intensity
            heatmap_surface.set_at((x, y), (*color, int(value * 0.3)))  # Color with 30% max opacity
    return heatmap_surface

def visualize_csv(tar_fname, csv_fname, predictions_dir, predictions_D_dir):
    baseline_predictions = {}
    duration_predictions = {}

    if SHOW_BASELINE:
        for filename in os.listdir(predictions_dir):
            if filename.startswith("gaze_") and filename.endswith(".png"):
                frame_id = filename[5:-4]
                pred_path = os.path.join(predictions_dir, filename)
                pred_heatmap = pygame.image.load(pred_path)
                baseline_predictions[frame_id] = pred_heatmap

    if SHOW_DURATIONS:
        for filename in os.listdir(predictions_D_dir):
            if filename.startswith("gaze_") and filename.endswith(".png"):
                parts = filename[:-4].split('_')  # Remove '.png' and split
                frame_id = '_'.join(parts[2:])  # Join all parts after the duration
                pred_path = os.path.join(predictions_D_dir, filename)
                pred_heatmap = pygame.image.load(pred_path)
                duration_predictions[frame_id] = {
                    'heatmap': pred_heatmap,
                    'predicted_duration': parts[1]  # The duration is the second part
                }

    frameid2pos, _, frameid2duration, _, _, _, frameid_list = data_reader.read_gaze_data_csv_file(csv_fname)
    frameid_list = sorted(frameid_list, key=frameid_sort_key)

    tar = tarfile.open(tar_fname, 'r')
    png_files = tar.getnames()

    preprocess_and_sanity_check(png_files, frameid_list)

    temp_extract_dir = os.path.dirname(tar_fname) + "/gaze_data_tmp/"
    if not os.path.exists(temp_extract_dir):
        os.mkdir(temp_extract_dir)
    tar.extractall(temp_extract_dir)
    temp_extract_full_path_dir = temp_extract_dir + '/' + png_files[0].split('/')[0]

    origin_w, origin_h = 160, 210
    x_scale, y_scale = 8.0, 4.0
    w, h = int(origin_w * x_scale), int(origin_h * y_scale)

    global ds
    ds.target_fps = 60
    ds.total_frame = len(png_files)
    ds.cur_frame_index = 0
    ds.terminated = False

    dw = DrawgcWrapper()

    pygame.init()
    pygame.font.init()
    pygame_font = pygame.font.SysFont('Consolas', 28)
    pygame.display.set_mode((w, h), pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.RLEACCEL, 32)
    screen = pygame.display.get_surface()

    while not ds.terminated:
        event_handler_func()
        
        frame_id = frameid_list[ds.cur_frame_index]
        png_fname = temp_extract_full_path_dir + '/' + frame_id + '.png'
        
        if not os.path.isfile(png_fname):
            screen.fill((0, 0, 0))
            text_surface_desc = pygame_font.render('Missing png file for frame id:', True, (255, 255, 255))
            screen.blit(text_surface_desc, (w // 10, 2 * h // 5))
            text_surface_frameid = pygame_font.render(frame_id, True, (255, 255, 255))
            screen.blit(text_surface_frameid, (w // 10, 3 * h // 5))
        else:
            s = pygame.image.load(png_fname)
            s = pygame.transform.scale(s, (w, h))
            screen.blit(s, (0, 0))

            gaze_list = frameid2pos[frame_id]
            if gaze_list is not None and len(gaze_list) > 0:
                for (posX, posY) in gaze_list:
                    if check_gaze_range(posX, posY, origin_w, origin_h):
                        dw.draw_gc(screen, (posX*x_scale, posY*y_scale))
            
            if SHOW_BASELINE and frame_id in baseline_predictions:
                baseline_heatmap = create_heatmap_surface(baseline_predictions[frame_id], (w, h), (0, 255, 0))
                screen.blit(baseline_heatmap, (0, 0))
            
            if SHOW_DURATIONS and frame_id in duration_predictions:
                duration_heatmap = create_heatmap_surface(duration_predictions[frame_id]['heatmap'], (w, h), (255, 0, 0))
                screen.blit(duration_heatmap, (0, 0))
                
                # Display duration labels
                ground_truth_duration = frameid2duration[frame_id]
                predicted_duration = duration_predictions[frame_id]['predicted_duration']
                
                gt_text = f"Ground Truth Duration: {ground_truth_duration}"
                pred_text = f"Predicted Duration: {predicted_duration}"
                
                gt_surface = pygame_font.render(gt_text, True, (255, 255, 255))
                pred_surface = pygame_font.render(pred_text, True, (255, 255, 255))
                
                screen.blit(gt_surface, (10, h - 70))
                screen.blit(pred_surface, (10, h - 35))

        pygame.display.flip()

        if not ds.pause:
            ds.cur_frame_index = (ds.cur_frame_index + 1) % ds.total_frame

        duration = frameid2duration[frame_id]
        if duration is not None:
            time.sleep(duration * 0.0000001)  # duration is in msec

    print("Replay ended.")
    shutil.rmtree(temp_extract_full_path_dir)
    
    

if __name__ == '__main__':
    
    tar_fname = "./ms_pacman/test.tar.bz2"
    csv_fname = "./ms_pacman/test.txt"
    predictions_dir = "./ms_pacman/predictions"  # Directory containing baseline predicted gaze heatmaps
    predictions_D_dir = "./ms_pacman/predictions_D"  # Directory containing duration-based predicted gaze heatmaps
    
    visualize_csv(tar_fname, csv_fname, predictions_dir, predictions_D_dir)
    






