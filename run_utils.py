from typing import List

import cv2
import norfair
import numpy as np
import pandas as pd
from norfair import Detection
from norfair.camera_motion import MotionEstimator

from inference import BaseClassifier, BaseDetector, Converter, YoloV5
from soccer import Ball, Match


def get_ball_detections(
    ball_detector: BaseDetector, frame: np.ndarray
) -> List[norfair.Detection]:
    ball_df = ball_detector.predict(frame)
    return Converter.DataFrame_to_Detections(ball_df)


def get_player_detections(
    person_detector: BaseDetector, classifier: BaseClassifier, frame: np.ndarray
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
