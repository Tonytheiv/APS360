from data import UCYDataset, ToTensor
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort 
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['WANDB_MODE'] = 'dryrun'  # Disables wandb

# Define paths
CSV_DIR = 'path/to/csv/dir'

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the specific YOLOv8 model you want to use
model.model.train()  # Set the model to training mode

# Define optimizer
optimizer = optim.Adam(model.model.parameters(), lr=0.001)

# Initialize DataLoader
ucy_dataset = UCYDataset(csv_dir=CSV_DIR, transform=ToTensor())
dataloader = DataLoader(ucy_dataset, batch_size=4, shuffle=True, num_workers=4)

# Initialize Deep SORT
tracker = DeepSort(max_age=30, n_init=3)

def visualize_image(img, tracker):
    plt.figure(figsize=(10, 10))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()
        track_id = track.track_id
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(img, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Training loop
def train(model, dataloader, optimizer, tracker, epochs=10):
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        for i_batch, sample_batched in enumerate(tqdm(dataloader, desc="Batches")):
            optimizer.zero_grad()

            for sample in sample_batched['data']:
                data = sample.numpy()
                for frame_data in data:
                    frame_number = frame_data[0]
                    pedestrian_id = frame_data[1]
                    x, y = frame_data[2], frame_data[3]

                    # Dummy image creation for testing (replace with actual video frame extraction)
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

                    # Perform pedestrian detection using YOLO
                    results = model.predict(img)  # Use predict method
                    boxes = results.pandas().xyxy[0]  # Get bounding boxes

                    # Update tracker
                    tracker.update_tracks(boxes)

                    # Visualize the detections and tracked IDs
                    visualize_image(img, tracker)

            # Dummy loss calculation for demonstration (replace with actual loss calculation)
            loss = torch.tensor(0.0)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs} completed.')

# Run training
train(model, dataloader, optimizer, tracker, epochs=10)
