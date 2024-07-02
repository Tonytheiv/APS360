import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class UCYDataset(Dataset):
    def __init__(self, video_dir, label_dir, transform=None):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.transform = transform
        self.videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        assert len(self.videos) == len(self.labels), "Mismatch between videos and labels"
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.videos[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        
        # Load labels
        labels = np.loadtxt(label_path, delimiter=',')
        
        sample = {'video': frames, 'labels': labels}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        video, labels = sample['video'], sample['labels']

        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        video = video.transpose((0, 3, 1, 2))
        return {'video': torch.from_numpy(video).float(),
                'labels': torch.from_numpy(labels).float()}
