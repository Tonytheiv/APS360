##########################################################################################################
'''
Author: Tony Wang
'''
##########################################################################################################

import os
import torch
import argparse
import yaml
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ultralytics import YOLO
from data import PedestrianTrackingDataset, collate_fn

FREQ = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for pedestrian tracking")
    parser.add_argument('--config', type=str, default='/home/tony/APS360/config/default.yaml', help='Path to the YAML config file')
    parser.add_argument('--data', type=str, default='/home/tony/APS360/config/custom_data.yaml', help='Path to the dataset YAML file')
    parser.add_argument('--input_dir', type=str, default='/home/tony/APS360/src/datasets/yolo_ucy', help='Path to output directory')
    parser.add_argument('--output_dir', type=str, default='/home/tony/APS360/output', help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--log_dir', type=str, help='Directory to save TensorBoard logs')
    parser.add_argument('--weights', type=str, help='Path to YOLOv8 weights')
    parser.add_argument('--dry_run', action='store_true', help='Enable dry run to avoid wandb')
    return parser.parse_args()

def override_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    return config

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = override_config_with_args(config, args)

    if args.dry_run:
        os.environ["WANDB_DISABLED"] = "true"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = PedestrianTrackingDataset(
        video_dir=os.path.join(args.input_dir, 'train', 'images'),
        label_dir=os.path.join(args.input_dir, 'train', 'labels'),
        output_dir=args.input_dir,
        phase='train',
        transform=transform
    )
    val_dataset = PedestrianTrackingDataset(
        video_dir=os.path.join(args.input_dir, 'val', 'images'),
        label_dir=os.path.join(args.input_dir, 'val', 'labels'),
        output_dir=args.input_dir,
        phase='val',
        transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = YOLO(config['weights'])
    
    model.data = args.data

    writer = SummaryWriter(log_dir=config['log_dir'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config["epochs"]}')):
            labels = [{'boxes': l[:, 1:], 'labels': torch.ones(len(l), dtype=torch.int64)} for l in labels]
            loss = model(images, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = running_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {avg_loss:.4f}')

        save_path = os.path.join(config['output_dir'], f'yolov8_epoch{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to: {save_path}")

    writer.close()

if __name__ == '__main__':
    main()
