import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy import interpolate

class UCYDataset(Dataset):
    def __init__(self, csv_dir, transform=None):
        self.csv_dir = csv_dir
        self.csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        self.transform = transform
        
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
        data = pd.read_csv(csv_path, delimiter=',', header=None).values
        interpolated_data = self.interpolate_missing(data)
        filtered_data = self.filter_frames(interpolated_data)
        
        sample = {'data': filtered_data}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def interpolate_missing(self, data):
        interpolated_data = np.empty((0, 4))
        
        max_id = int(np.max(data[:, 1]) + 1)
        for i in range(max_id):
            traj_of_ped_i = data[data[:, 1] == i, :]
            if traj_of_ped_i.size == 0:
                continue
            y = int(traj_of_ped_i[-1, 0])
            x = int(traj_of_ped_i[0, 0])
            
            while x <= y:
                if x in traj_of_ped_i[:, 0]:
                    interpolated_data = np.append(interpolated_data, traj_of_ped_i[traj_of_ped_i[:, 0] == x], axis=0)
                    x += 1
                else:
                    f = interpolate.interp1d(traj_of_ped_i[:, 0], [traj_of_ped_i[:, 2], traj_of_ped_i[:, 3]], 
                                             fill_value="extrapolate", bounds_error=False)
                    inter = f(x)
                    interpolated_data = np.append(interpolated_data, np.array([[int(x), int(i), float(inter[0]), float(inter[1])]]), axis=0)
                    x += 1
                if x == y + 1:
                    break
        interpolated_data = interpolated_data[np.argsort(interpolated_data[:, 0])]
        return interpolated_data
    
    def filter_frames(self, data, frame_interval=10):
        frames = sorted(np.unique(data[:, 0]))
        final_data = []
        for frame in frames:
            if frame % 10 == 0 or frame == 0:
                final_data.extend(data[data[:, 0] == frame, :])
        final_data = np.unique(final_data, axis=0)
        return np.array(final_data)

class ToTensor(object):
    def __call__(self, sample):
        data = sample['data']
        return {'data': torch.from_numpy(data).float()}
