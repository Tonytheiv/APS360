#!/bin/bash

mkdir -p /home/tony/APS360/output
mkdir -p /home/tony/APS360/src/runs

python3 /home/tony/APS360/src/model.py \
                --data custom_data.yaml \
                --video_dir "/home/tony/APS360/baseline/UCY_dataset" \
                --label_dir "/home/tony/APS360/baseline/processed_data" \
                --batch_size 4 \
                --epochs 50 \
                --log_dir "/home/tony/APS360/src/runs" \
                --weights "/home/tony/APS360/src/yolov8n.pt" \
                --output_dir "/home/tony/APS360/output" \
                --dry_run