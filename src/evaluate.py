##########################################################################################################
'''
Author: Tony Wang
'''
##########################################################################################################

import argparse
import pandas as pd
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions by comparing with ground truth")
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to the ground truth CSV file')
    parser.add_argument('--predictions', type=str, required=True, help='Path to the predictions CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the evaluation results')
    return parser.parse_args()

def calculate_error(gt_df, pred_df):
    """
    Calculates the error for each pedestrian ID as the Euclidean distance between the centroids of the bounding boxes.

    Args:
        gt_df (DataFrame): Ground truth DataFrame
        pred_df (DataFrame): Predictions DataFrame
    
    Returns:
        DataFrame: DataFrame containing errors for each pedestrian ID
    """
    errors = []
    for pid in gt_df['PedestrianID'].unique():
        gt_pid_df = gt_df[gt_df['PedestrianID'] == pid]
        pred_pid_df = pred_df[pred_df['PedestrianID'] == pid]
        
        if not pred_pid_df.empty:
            merged_df = pd.merge(gt_pid_df, pred_pid_df, on='FrameID', suffixes=('_gt', '_pred'))
            merged_df['error'] = np.sqrt((merged_df['PosX_gt'] - merged_df['PosX_pred'])**2 + 
                                         (merged_df['PosY_gt'] - merged_df['PosY_pred'])**2)
            errors.append(merged_df[['FrameID', 'PedestrianID_gt', 'error']].rename(columns={'PedestrianID_gt': 'PedestrianID'}))
    
    if errors:
        return pd.concat(errors, ignore_index=True)
    else:
        return pd.DataFrame(columns=['FrameID', 'PedestrianID', 'error'])

def main():
    args = parse_args()

    gt_df = pd.read_csv(args.ground_truth)
    pred_df = pd.read_csv(args.predictions)
    error_df = calculate_error(gt_df, pred_df)
    error_df.to_csv(args.output_file, index=False)
    print(f'Error evaluation saved to {args.output_file}')

if __name__ == '__main__':
    main()
