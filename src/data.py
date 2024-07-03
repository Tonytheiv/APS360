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

class PedestrianTrackingDataset(Dataset):
    def __init__(self, video_dir, label_dir, image_dir, transform=None):
        """
        Initializes the dataset with directories for videos, labels, and images.

        Args:
            video_dir (str): Directory containing video files.
            label_dir (str): Directory containing label CSV files.
            image_dir (str): Directory to store extracted images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.video_files = [f for f in os.listdir(video_dir) if f[0].isdigit() and f.endswith('.avi')]
        self.video_files.sort()
        
        self._prepare_images()
        
    def _prepare_images(self):
        """
        Extracts frames from video files and saves them as JPEG images in the image directory.
        This function is only run if the images do not already exist.
        """
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
        for video_file in tqdm(self.video_files, desc='Extracting frames'):
            video_path = os.path.join(self.video_dir, video_file)
            video_id = video_file.split('.')[0]

            cap = cv2.VideoCapture(video_path)
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_name = f'{video_id}_frame{frame_id}.jpg'
                image_path = os.path.join(self.image_dir, image_name)
                
                if not os.path.exists(image_path):
                    cv2.imwrite(image_path, frame)
                
                frame_id += 1
            cap.release()

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
        video_id = image_file.split('_frame')[0]
        frame_id = int(image_file.split('_frame')[1].replace('.jpg', ''))
        
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        
        label_file = os.path.join(self.label_dir, f'{video_id}.csv')
        labels_df = pd.read_csv(label_file, header=None)
        labels_df.columns = ['FrameID', 'PedestrianID', 'PosX', 'PosY']
        
        labels = labels_df[labels_df['FrameID'] == frame_id][['PedestrianID', 'PosX', 'PosY']].values
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
    image_dir = '/home/tony/APS360/src/datasets/ucy'
    
    dataset = PedestrianTrackingDataset(video_dir, label_dir, image_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx}:')
        print('Images shape:', images.shape)
        print('Labels:', labels)
        break
