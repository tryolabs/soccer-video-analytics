import cv2
import numpy as np
from norfair import AbsolutePaths, Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import iou, mean_euclidean

from inference import Converter, HSVClassifier, NNClassifier, YoloV5, hsv_classifier
from inference.filters import filters
from inference.inertia_classifier import InertiaClassifier
from run_utils import (
    classify_city_gk,
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath

video = Video(input_path="videos/soccer_posession.mp4")
# video = Video(input_path="videos/soccer_posession_2.mp4")
# video = Video(input_path="videos/soccer_posession_short.mp4")

# Object Detectors
player_detector = YoloV5()
ball_detector = YoloV5(model_path="models/best.pt")


def get_points_to_draw(points: np.array) -> np.ndarray:
    xmin, ymin = points[0]
    xmax, ymax = points[1]

    return np.array([[(xmin + xmax) / 2, ymax]])


ball_path_drawer = AbsolutePaths(max_history=50, thickness=2, color=(255, 255, 255))
player_path_drawer = AbsolutePaths(
    max_history=4, thickness=2, get_points_to_draw=get_points_to_draw
)

# Classifier
classifier = NNClassifier(
    model_path="models/model_classification.pt",
    classes=["Chelsea", "Man City", "Referee"],
)


hsv_classifier = HSVClassifier(filters=filters)
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
man_city = Team(name="Man City", abbreviation="MNC", color=(240, 230, 188))
chelsea = Team(
    name="Chelsea",
    abbreviation="CHE",
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
teams = [man_city, chelsea]
match = Match(home=chelsea, away=man_city)
match.team_possession = man_city

# Tracking
DISTANCE_THRESHOLD_BBOX: float = 0.65
DISTANCE_THRESHOLD_CENTROID: int = 200

player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=DISTANCE_THRESHOLD_CENTROID,
    initialization_delay=10,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None


# past_points = []
path = AbsolutePath()

for i, frame in enumerate(video):

    # if i > 400:
    #     continue

    # Get Detections
    players_detections = get_player_detections(player_detector, classifier, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    for i, player in enumerate(player_detections):
        player_detections[i].data["label"] = player.data["id"]

    for i, a_ball in enumerate(ball_detections):
        ball_detections[i].data["label"] = a_ball.data["id"]

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    classify_city_gk(player_detections)

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball)

    # Draw
    ball_detection = None if ball is None else ball.detection

    frame = path.draw(
        img=frame,
        detection=ball_detection,
        coord_transformations=coord_transformations,
        color=match.team_possession.color,
    )

    # frame = Player.draw_players(players=players, frame=frame, confidence=False)

    # if ball:
    #     frame = ball.draw(frame)

    frame = match.draw(frame, debug=False)
    # frame = player_path_drawer.draw(
    #     frame, player_track_objects, coord_transform=coord_transformations
    # )
    # frame = ball_path_drawer.draw(
    #     frame, ball_track_objects, coord_transform=coord_transformations
    # )

    # Write video
    video.write(frame)
    # video.show(frame)
