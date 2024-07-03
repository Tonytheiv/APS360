##########################################################################################################
'''
Author: Jonathan Choi, Tony Wang
'''
##########################################################################################################

import os
import math
import cv2
import numpy as np
from numpy import savetxt
from scipy import interpolate

class PedestrianDataProcessor:
    def __init__(self, data_dir, output_dir, video_filename, raw_data_filename):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.video_path = os.path.join(self.data_dir, video_filename)
        self.raw_data_path = os.path.join(self.data_dir, raw_data_filename)
        self.cap = cv2.VideoCapture(self.video_path)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video.")
        else:
            _, frame = self.cap.read()
            self.height = int(frame.shape[0])
            self.width = int(frame.shape[1])

    def coordinate_transformation(self, x_c, y_c):
        cx = self.width // 2
        cy = self.height // 2
        x_cv = x_c + cx
        y_cv = -y_c + cy
        return [x_cv, y_cv]

    def load_and_transform_data(self):
        with open(self.raw_data_path, 'r') as file:
            num_pedestrians = int(file.readline().split(' - ')[0])
            num_observations = len(file.readlines()) - num_pedestrians
        
        data = np.zeros((num_observations, 4))
        with open(self.raw_data_path, 'r') as file:
            file.readline()
            row_count = 0
            for i in range(num_pedestrians):
                num_control_points_i = int(file.readline().split(' - ')[0])
                for j in range(num_control_points_i):
                    points = file.readline().split(' ')
                    data[row_count, 0:4] = [int(points[2]), i] + self.coordinate_transformation(float(points[0]), float(points[1]))
                    row_count += 1
        
        sorted_indices = np.argsort(data[:, 0])
        sorted_data = data[sorted_indices]
        savetxt(os.path.join(self.output_dir, 'Data_formated_crowds_zara02.csv'), sorted_data, delimiter=',')
        return sorted_data

    def interpolate_data(self, sorted_data):
        data_to_interpolate = np.zeros((len(sorted_data), 4), dtype=float)
        for i in range(len(sorted_data)):
            data_to_interpolate[i, 0] = sorted_data[i, 0]
            data_to_interpolate[i, 1] = sorted_data[i, 1]
            data_to_interpolate[i, 2] = sorted_data[i, 2]
            data_to_interpolate[i, 3] = sorted_data[i, 3]

        interpolated_data = np.empty((0, 4), dtype=float)
        t = 0
        for i in range(1, int(np.max(data_to_interpolate[:, 1]) + 1)):
            mask = data_to_interpolate[:, 1] == i
            traj_of_ped_i = data_to_interpolate[mask, :]
            if traj_of_ped_i.size == 0:
                print('-----------------------------------------------------')
                print(f'This PedID does not exist in the data: {i}')
                t += 1
            else:
                x = int(traj_of_ped_i[0, 0])
                y = int(traj_of_ped_i[-1, 0])

                if x % 10 != 0:
                    if x % 10 < 5:
                        x -= x % 10
                    else:
                        x += 10 - x % 10
                
                if y % 10 != 0:
                    if y % 10 < 5:
                        y -= y % 10
                    else:
                        y += 10 - y % 10

                while x < y:
                    for j in range(traj_of_ped_i.shape[0]):
                        z = np.where(traj_of_ped_i[:, 0] == x)
                        if np.squeeze(traj_of_ped_i[z, 0]) == x:
                            exist_frame = traj_of_ped_i[z, :]
                            interpolated_data = np.append(interpolated_data, exist_frame[0, :, :], axis=0)
                            x += 1
                        else:
                            f = interpolate.interp1d(traj_of_ped_i[:, 0], traj_of_ped_i[:, 2:4].T, fill_value="extrapolate", bounds_error=False)
                            inter = f(x)
                            interpolated_data = np.append(interpolated_data, np.array([[x, i, inter[0], inter[1]]]), axis=0)
                            x += 1
                        if x == y + 1:
                            break

            percentage = i / int(np.max(data_to_interpolate[:, 1]) + 1) * 100
            percentage = "{:.2f}".format(percentage)
            print(f'Interpolation percentage: {percentage}%')
            print('-----------------------------------------------------')

        print(f'Number of missing pedestrians is: {t}')
        print('-----------------------------------------------------')

        interpolated_data = interpolated_data[np.argsort(interpolated_data[:, 0])]
        savetxt(os.path.join(self.output_dir, 'interpolated_data_crowds_zara02.csv'), interpolated_data, delimiter=',')

if __name__ == '__main__':
    data_processor = PedestrianDataProcessor(
        data_dir='~/APS360/baseline/UCY_dataset',
        output_dir='~/APS360/baseline/processed_data',
        video_filename='crowds_zara02.avi',
        raw_data_filename='raw_zara02.txt'
    )

    sorted_data = data_processor.load_and_transform_data()
    data_processor.interpolate_data(sorted_data)
