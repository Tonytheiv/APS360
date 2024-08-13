import os
import random

import cv2
from ultralytics import YOLO
import torch

from tracker import Tracker

#sample video path
# video_path = os.path.join('.', 'data', 'clip.mp4')
video_path = os.path.join('.', 'data', 'crowd_clip.avi')
# video_path = os.path.join('.', 'data', 'zara02.mp4')
# video_path = os.path.join('.', 'data', '203_college_straight.mov')
# video_path = os.path.join('.', 'data', 'crowds_zara02.avi')
# video_path = os.path.join('.', 'data', 'mii.mp4')

#output video path
# video_out_path = os.path.join('.', 'clip_m.mp4')
video_out_path = os.path.join('.', 'zara_yolo.mp4')
# video_out_path = os.path.join('.', 'college_m.mp4')
# video_out_path = os.path.join('.', 'test.mp4')

# log = open(os.path.join('.', "zara_m7.txt"), "w")
log = open(os.path.join('.', "zara_yolo.txt"), "w")
# log = open(os.path.join('.', "college_m.txt"), "w")
# log = open(os.path.join('.', "test.txt"), "w")

#Video capture into CV2
capture = cv2.VideoCapture(video_path)

more_frames, frame = capture.read()

capture_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), capture.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("best.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]

detection_threshold = 0.2
frame_counter = 0
while more_frames:
    results = model(frame)

    for result in results:
        detections = []
        for res in result.boxes.data.tolist():
            x_1, y_1, x_2, y_2, probability, class_id = res
            x_1 = int(x_1)
            x_2 = int(x_2)
            y_1 = int(y_1)
            y_2 = int(y_2)
            class_id = int(class_id)
            if probability > detection_threshold and class_id == 0:
                detections.append([x_1, y_1, x_2, y_2, probability])
                # cv2.rectangle(frame, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 0, 255), 3)

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x_1, y_1, x_2, y_2 = bbox
            track_id = track.track_id

            head_coords = (int((x_1+x_2)/2), int((y_1+y_2)/2-(y_2-y_1)/4))

            cv2.rectangle(frame, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (colors[track_id % len(colors)]), 3)
            cv2.circle(frame, head_coords, 2, (colors[track_id % len(colors)]), 3)
            log.write(f'{frame_counter},{track_id},{head_coords[0]},{head_coords[1]}\n')

    capture_out.write(frame)
    more_frames, frame = capture.read()
    frame_counter += 1

capture.release()
capture_out.release()
cv2.destroyAllWindows()