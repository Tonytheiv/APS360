# APS360

## Baseline

In order to run the baseline you must run the `preprocess.ipynb` first, running all cells should work.

(Currently only crowds_zara02.avi are present in the data folder)

Running `baseline.ipynb` will use cv2 centroid tracking and frame differencing to identify moving elements of the video. The program will output a processed video in `processsed_videos`.

### Info

The `maxDissapeared` argument in `CentroidTracker()` will change how long objects persist in "memory." Line 84 in cell 1 in `baseline.ipynb` sets this to 0; the default is 50.

## YOLOv8 Pedestrian Tracking

### Overview

This project fine-tunes the YOLOv8 model for pedestrian tracking. It includes scripts to preprocess the data, train the model, and visualize the results.

### Project Structure
APS360

├── config/

│ ├── custom_data.yaml

│ ├── default.yaml

├── launch/

│ ├── train.sh

├── baseline/

│ ├── UCY_dataset/

│ ├── processed_data/

├── src/

│ ├── data.py

│ ├── model.py

│ ├── datasets/

│ │ ├── yolo_ucy/

### Requirements

- Python 3.8+
- PyTorch 2.0.1+cu118
- CUDA (compatible GPU and drivers)
- Ultralytics YOLOv8
- OpenCV
- scikit-learn
- TensorBoard
- tqdm
- PyYAML

### Installation

1. **Clone the repository**:

   ```bash
   git clone git@github.com:Tonytheiv/APS360.git
   cd APS360
   ```
2. **Create a python environment**
  ```bash
python3 -m venv venv
source venv/bin/activate
```
3. **Install dependencies**
```bash
pip install torch torchvision torchaudio
pip install opencv-python-headless
pip install scikit-learn
pip install tensorboard
pip install tqdm
pip install pyyaml
pip install ultralytics
```
### Configuration

1. **Prepare the data**:

   Make sure you have your videos and CSV files placed correctly:

   - Videos should be in `baseline/UCY_dataset/`
   - CSV files should be in `baseline/processed_data/`

2. **Set up the configuration files**:

   `custom_data.yaml` (example):

   ```yaml
   train: /home/tony/APS360/src/datasets/yolo_ucy/train/images
   val: /home/tony/APS360/src/datasets/yolo_ucy/val/images
   test: /home/tony/APS360/src/datasets/yolo_ucy/test/images

   nc: 1  # Number of classes
   names: ['pedestrian']  # Class names
      `default.yaml` (example):

   ```yaml
   batch_size: 16
   epochs: 100
   log_dir: /home/tony/APS360/runs
   weights: /home/tony/APS360/src/yolov8n.pt

   
