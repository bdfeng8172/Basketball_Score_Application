from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
import torch
import numpy as np

class PlayerTracker:
    def __init__(self):
        default_args = SimpleNamespace(
            track_thresh=0.3,
            track_buffer=30,
            match_thresh=0.8,
            aspect_ratio_thresh=1.6,
            min_box_area=10,
            mot20=False
        )
        self.tracker = BYTETracker(default_args)
        self.tracked_ids = set()

    def update(self, detections, img_info, img_size):


            # Ensure detections is a numpy array
            # if not isinstance(detections, np.ndarray):
            #     detections = np.array(detections)

            # if not isinstance(detections, torch.Tensor):
            #     detections = torch.from_numpy(detections).float().to('cuda')


        online_targets = self.tracker.update(
            detections,
            img_info,
            img_size
        )

        tracked_players = []
        for t in online_targets:
            tid = int(t.track_id)
            x1, y1, x2, y2 = map(int, t.tlbr)
            tracked_players.append((tid, x1, y1, x2, y2))
        return tracked_players
