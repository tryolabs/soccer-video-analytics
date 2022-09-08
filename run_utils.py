from typing import List, Tuple

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
    person_detections = Converter.DataFrame_to_Detections(person_df)
    return person_detections
    # return classifier.predict_from_detections(detections=person_detections, img=frame)
    # person_df = classifier.predict_from_df(df=person_df, img=frame)
    # return Converter.DataFrame_to_Detections(person_df)


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


players_teams = {}
classification_counter = 40


def should_classify(detection: Detection) -> bool:
    return True

    # if detection.data["id"] not in players_teams:
    #     return True
    # elif len(players_teams[detection.data["id"]]) < classification_counter:
    #     return True
    # else:
    #     return False


def add_window_classification(detection: Detection):
    global players_teams
    if detection.data["id"] not in players_teams:
        players_teams[detection.data["id"]] = [detection.data["classification"]]
    elif len(players_teams[detection.data["id"]]) < classification_counter:
        players_teams[detection.data["id"]].append(detection.data["classification"])
    elif len(players_teams[detection.data["id"]]) == classification_counter:
        players_teams[detection.data["id"]].pop(0)
        players_teams[detection.data["id"]].append(detection.data["classification"])


def add_first_n_classification(detection: Detection):
    global players_teams

    if detection.data["id"] not in players_teams:
        players_teams[detection.data["id"]] = [detection.data["classification"]]
    elif len(players_teams[detection.data["id"]]) < classification_counter:
        players_teams[detection.data["id"]].append(detection.data["classification"])


def add_new_clasifications_to_players_teams(detections: List[Detection]):
    global players_teams

    for detection in detections:
        add_window_classification(detection)
        # add_first_n_classification(detection)


def set_detections_classification(detections: List[Detection]):
    global players_teams
    for detection in detections:
        previous_classifications = players_teams[detection.data["id"]]
        detection.data["classification"] = max(
            set(previous_classifications), key=previous_classifications.count
        )


def classify_city_gk(detections: List[Detection]):
    referee_detections = [
        detection
        for detection in detections
        if detection.data["classification"] == "Referee"
    ]

    if len(referee_detections) == 2:
        # get the detection at the left
        city_gk_detection = min(referee_detections, key=lambda x: x.points[0])
        city_gk_detection.data["classification"] = "Man City"
