# TAGIL: Temporal Attention Guided Imitation Learning

## Paper Reference
The methodology for this code implementation can be found in: `TAGIL.pdf`

## Dataset
Game gaze data can be downloaded from the Atari-HEAD dataset:  
[Zenodo - Atari-HEAD Dataset](https://zenodo.org/records/2603190)

## Setup and Installation

### Important Note
Before running the code, ensure you either:
- Update the Weights & Biases login key to your personal key
- Comment out the disable line to prevent logging to Weights & Biases

## Running the Code

### 1. Preprocessing

#### Combining Multiple Games
If you want to train on multiple gameplay sequences simultaneously run concat.py 
and enter the parent directory that contains all the pairs of image and .txt files.

This will concatenate multiple gameplays into a single `.tar.bz2` file.

#### Required Files
Both Gaze and T-Gaze models require 4 input files:
- Original dataset files:
  - `.txt` file
  - `.tar.bz2` file
- Preprocessed files:
  - Optical flow file
  - Saliency file

To generate the preprocessed files:
1. Edit the file paths in `optical_flow.py` and run it
2. Edit the file paths in `saliency.py` and run it

**Note**: This preprocessing must be performed separately for train, validation, and test files.

### 2. Gaze Prediction

#### Training
1. Configure file paths in either:
   - `Gaze_baseline.py`
   - `t_gaze.py`
2. Run the chosen script to train the model

#### Inference
1. Use `inference.py` with your trained model to generate `.npz` files
2. Visualize predictions using `show_pred.py`

#### Game Replay Visualization
Use `gaze_replay.py` to watch the game replay with gaze predictions overlay:
- Set boolean variables at the top of the file to choose visualization options:
  - Baseline predictions
  - T-gaze predictions
  - Or both simultaneously

### 3. Action Prediction
1. Prepare the following files:
   - `.txt` file
   - `.tar.bz2` file
   - `.npz` file (generated from gaze prediction)
2. Run either:
   - `agil.py`
   - `tagil.py`
3. Edit the relevant file paths in your chosen script

The script will train and produce an action prediction model.
