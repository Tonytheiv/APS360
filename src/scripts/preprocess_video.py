##########################################################################################################
'''
Author: Tony Wang
'''
##########################################################################################################

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)
from preprocess.interpolate import PedestrianDataProcessor


data_processor = PedestrianDataProcessor(
    data_dir='/home/tony/APS360/baseline/UCY_dataset',
    output_dir='/home/tony/APS360/baseline/processed_data',
    video_filename='/home/tony/APS360/baseline/UCY_dataset/crowds_zara02.avi',
    raw_data_filename='/home/tony/APS360/baseline/UCY_dataset/raw_zara02.txt'
)

sorted_data = data_processor.load_and_transform_data()
data_processor.interpolate_data(sorted_data)