##########################################################################################################
'''
Author: Tony Wang
'''
##########################################################################################################

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class PedestrianTrackingDataset(Dataset):
    def __init__(self, video_dir, label_dir, output_dir, phase='train', transform=None, train_split=0.7, val_split=0.2, test_split=0.1, random_state=42):
        """
        Initializes the dataset with directories for videos, labels, and output images.

        Args:
            video_dir (str): Directory containing video files.
            label_dir (str): Directory containing label CSV files.
            output_dir (str): Directory to store extracted images and labels in YOLO format.
            phase (str): 'train', 'val' or 'test' to specify the phase.
            transform (callable, optional): A function/transform to apply to the images.
            train_split (float): Ratio of training data to total data.
            val_split (float): Ratio of validation data to total data.
            test_split (float): Ratio of test data to total data.
            random_state (int): Seed for random split.
        """
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.phase = phase
        self.transform = transform
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.image_dir = os.path.join(output_dir, phase, 'images')
        self.label_dir_out = os.path.join(output_dir, phase, 'labels')
        self.visualization_dir = os.path.join(output_dir, phase, 'visualizations')

        if not os.path.exists(self.image_dir) or len(os.listdir(self.image_dir)) == 0:
            self._extract_frames()
            self._prepare_images()

    def _extract_frames(self):
        """
        Extracts frames from video files and saves them as JPEG images in the image directory.
        """
        os.makedirs(self.video_dir, exist_ok=True)

        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.avi')]
        for video_file in tqdm(video_files, desc='Extracting frames'):
            video_path = os.path.join(self.video_dir, video_file)
            video_id = video_file.split('.')[0]

            cap = cv2.VideoCapture(video_path)
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_name = f'{video_id}_frame{frame_id}.jpg'
                image_path = os.path.join(self.video_dir, image_name)

                if not os.path.exists(image_path):
                    cv2.imwrite(image_path, frame)
                
                frame_id += 1
            cap.release()

    def _prepare_images(self):
        """
        Organizes the extracted frames into train, val, and test directories and generates labels in YOLO format.
        """
        for dir_path in [self.image_dir, self.label_dir_out, self.visualization_dir]:
            os.makedirs(dir_path, exist_ok=True)

        image_files = [f for f in os.listdir(self.video_dir) if f.endswith('.jpg')]
        image_files.sort()
        
        train_files, temp_files = train_test_split(image_files, test_size=(self.val_split + self.test_split), random_state=self.random_state)
        val_files, test_files = train_test_split(temp_files, test_size=self.test_split / (self.val_split + self.test_split), random_state=self.random_state)

        phase_files = {'train': train_files, 'val': val_files, 'test': test_files}

        for phase, files in phase_files.items():
            for image_file in tqdm(files, desc=f'Processing {phase} images and labels'):
                image_path = os.path.join(self.video_dir, image_file)
                
                if phase == 'train':
                    dest_image_dir = os.path.join(self.output_dir, 'train', 'images')
                    dest_label_dir = os.path.join(self.output_dir, 'train', 'labels')
                    dest_vis_dir = os.path.join(self.output_dir, 'train', 'visualizations')
                elif phase == 'val':
                    dest_image_dir = os.path.join(self.output_dir, 'val', 'images')
                    dest_label_dir = os.path.join(self.output_dir, 'val', 'labels')
                    dest_vis_dir = os.path.join(self.output_dir, 'val', 'visualizations')
                else:
                    dest_image_dir = os.path.join(self.output_dir, 'test', 'images')
                    dest_label_dir = os.path.join(self.output_dir, 'test', 'labels')
                    dest_vis_dir = os.path.join(self.output_dir, 'test', 'visualizations')
                
                os.makedirs(dest_image_dir, exist_ok=True)
                os.makedirs(dest_label_dir, exist_ok=True)
                os.makedirs(dest_vis_dir, exist_ok=True)

                # Move image
                dest_image_path = os.path.join(dest_image_dir, image_file)
                if not os.path.exists(dest_image_path):
                    os.rename(image_path, dest_image_path)
                
                # Create label
                video_id, frame_id = image_file.split('_frame')
                frame_id = int(frame_id.replace('.jpg', ''))
                label_file = os.path.join(self.label_dir, f'{video_id}.csv')
                
                labels_df = pd.read_csv(label_file, header=None)
                labels_df.columns = ['FrameID', 'PedestrianID', 'PosX', 'PosY']
                
                frame_labels = labels_df[labels_df['FrameID'] == frame_id]
                label_txt = os.path.join(dest_label_dir, f'{video_id}_frame{frame_id}.txt')
                
                with open(label_txt, 'w') as f:
                    for _, row in frame_labels.iterrows():
                        x_center = row['PosX'] / 720  
                        y_center = row['PosY'] / 576  
                        width = height = 25 / 720  
                        f.write(f'0 {x_center} {y_center} {width} {height}\n')

                # Create visualization
                vis_image = cv2.imread(dest_image_path)
                for _, row in frame_labels.iterrows():
                    x_center = int(row['PosX'])
                    y_center = int(row['PosY'])
                    top_left = (x_center - 12, y_center - 12)
                    bottom_right = (x_center + 12, y_center + 12)
                    cv2.rectangle(vis_image, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(vis_image, f'ID: {int(row["PedestrianID"])}', (x_center - 10, y_center - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                vis_image_path = os.path.join(dest_vis_dir, image_file)
                cv2.imwrite(vis_image_path, vis_image)

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
    
    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding labels.

        Args:
            idx (int): Index of the image to retrieve.
        
        Returns:
            image (Tensor): The image tensor.
            labels (Tensor): Corresponding labels for the image.
        """
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        image_files.sort()
        image_file = image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)

        label_file = os.path.join(self.label_dir_out, f'{os.path.splitext(image_file)[0]}.txt')
        labels = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    labels.append([float(x) for x in parts])
        
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(labels, dtype=torch.float32)

def collate_fn(batch):
    """
    Custom collate function to handle batching of images and labels.

    Args:
        batch (list): List of tuples containing images and labels.
    
    Returns:
        images (Tensor): Batched images.
        labels (list): Batched labels.
    """
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, labels

if __name__ == "__main__":
    video_dir = '/home/tony/APS360/baseline/UCY_dataset'
    label_dir = '/home/tony/APS360/baseline/processed_data'
    output_dir = '/home/tony/APS360/src/datasets/yolo_ucy'
    
    for phase in ['train', 'val', 'test']:
        dataset = PedestrianTrackingDataset(video_dir, label_dir, output_dir, phase=phase)
    
    dataset = PedestrianTrackingDataset(video_dir, label_dir, output_dir, phase='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx}:')
        print('Images shape:', images.shape)
        print('Labels:', labels)
        break 

