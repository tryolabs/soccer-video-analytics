from typing import List

import cv2
import norfair
import numpy as np
import pandas as pd
from norfair import Detection
from norfair.camera_motion import MotionEstimator

from inference import BaseDetector, Classifier, Converter, YoloV5
from soccer import Ball, Match


def get_ball_detections(
    ball_detector: BaseDetector, frame: np.ndarray
) -> List[norfair.Detection]:
    ball_df = ball_detector.predict(frame)
    return Converter.DataFrame_to_Detections(ball_df)


def get_player_detections(
    person_detector: BaseDetector, classifier: Classifier, frame: np.ndarray
) -> List[norfair.Detection]:

    person_df = person_detector.predict(frame)
    person_df = person_df[person_df["name"] == "person"]
    person_df = person_df[person_df["confidence"] > 0.5]
    person_df = classifier.predict_from_df(df=person_df, img=frame)
    return Converter.DataFrame_to_Detections(person_df)


def create_mask(frame: np.ndarray, detections: List[norfair.Detection]) -> np.ndarray:
    """
    Apply mask to frame.
    """

    if not detections:
        mask = np.ones(frame.shape[:2], dtype=frame.dtype)
    else:
        detections_df = Converter.Detections_to_DataFrame(detections)
        mask = YoloV5.generate_predictions_mask(detections_df, frame)

    # remove dashboard counter
    mask[80:200, 160:500] = 0
    mask[58:230, 1400:1780] = 0

    return mask


def apply_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to frame.
    """
    masked_frame = frame.copy()
    masked_frame[mask == 0] = 0
    return masked_frame


def update_motion_estimator(
    motion_estimator: MotionEstimator,
    detections: List[Detection],
    frame: np.ndarray,
):
    mask = create_mask(frame=frame, detections=detections)
    coord_transformations = motion_estimator.update(frame, mask=mask)
    return coord_transformations


def _iou(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Underlying iou distance. See `Norfair.distances.iou`.
    """

    # Detection points will be box A
    # Tracked objects point will be box B.
    box_a = np.concatenate([detection.points[0], detection.points[1]])
    box_b = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and tracker
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [0, 1]
    return 1 - iou


def mean_euclidean(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Average euclidean distance between the points in detection and estimates in tracked_object.
    See `np.linalg.norm`.
    """
    return np.linalg.norm(detection.points - tracked_object.estimate, axis=1).mean()


def get_main_ball(detections: List[Detection], match: Match = None) -> Ball:
    """
    Returns the detection with the highest score.
    """
    if not detections:
        return None

    # main_ball_ids = [22, 77]

    # for detection in detections:
    #     if int(detection.data["id"]) in main_ball_ids:
    #         main_ball_detection = detection

    main_ball_detection = detections[0]

    ball = Ball(main_ball_detection)
    if match:
        ball.set_color(match)

    return ball
