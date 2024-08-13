from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.25
        # max_cosine_distance = 0.4
        # max_cosine_distance = 1
        budget = None

        # hardcoded filepath sorry
        encoder_model = '/home/ben/Downloads/Object_Tracking/object-tracking-yolov8-deep-sort/model_data/mars-small128.pb'

        distance_metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, budget)
        self.tracker = DeepSortTracker(distance_metric, n_init=2, max_age=60)
        self.encoder = gdet.create_box_encoder(encoder_model, batch_size=1)

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bounding_boxes = np.asarray([d[:-1] for d in detections], dtype=np.int32)
        bounding_boxes[:, 2:] = bounding_boxes[:, 2:] - bounding_boxes[:, 0:2]
        scores = np.array([d[-1] for d in detections], dtype=np.int32)

        features = self.encoder(frame, bounding_boxes)

        detects = []
        for bbox_id, bbox in enumerate(bounding_boxes):
            detects.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(detects)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            tracks.append(Track(id, bbox))

        self.tracks = tracks

class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
