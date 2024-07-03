import os
import cv2
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with YOLOv8 and output predictions")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv8 weights')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images with bounding boxes')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to save predictions CSV file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for YOLOv8 model')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold for predictions')
    return parser.parse_args()

def run_inference(args):
    model = YOLO(args.weights)
    os.makedirs(args.output_dir, exist_ok=True)
    
    predictions = []
    
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jpg')]
    for image_file in tqdm(image_files, desc='Running inference'):
        image_path = os.path.join(args.input_dir, image_file)
        image = cv2.imread(image_path)
        
        results = model.predict(source=image_path, imgsz=args.imgsz, conf=args.confidence)
        
        annotated_image = image.copy()
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            conf = result.conf[0]
            cls = result.cls[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, f'ID: {int(cls)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            frame_id = int(image_file.split('_frame')[1].split('.jpg')[0])
            pedestrian_id = int(cls) + 1  
            pos_x = (x1 + x2) / 2
            pos_y = (y1 + y2) / 2
            predictions.append([frame_id, pedestrian_id, pos_x, pos_y])
        
        output_image_path = os.path.join(args.output_dir, image_file)
        cv2.imwrite(output_image_path, annotated_image)
    
    predictions_df = pd.DataFrame(predictions, columns=['FrameID', 'PedestrianID', 'PosX', 'PosY'])
    predictions_df.to_csv(args.predictions_file, index=False)

if __name__ == '__main__':
    args = parse_args()
    run_inference(args)
