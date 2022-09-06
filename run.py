from norfair import AbsolutePaths, Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import iou_opt, mean_euclidean

from inference import Converter, NNClassifier, YoloV5
from inference.yolov5 import YoloV5
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team

DISTANCE_THRESHOLD_BBOX: float = 0.8
DISTANCE_THRESHOLD_CENTROID: int = 120
MAX_DISTANCE: int = 10000

video = Video(input_path="videos/soccer_posession_short.mp4")

player_tracker = Tracker(
    distance_function=iou_opt,
    distance_threshold=DISTANCE_THRESHOLD_BBOX,
    initialization_delay=3,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=200,
    initialization_delay=10,
    hit_counter_max=2000,
)

path_drawer = AbsolutePaths(max_history=50, thickness=2, color=(255, 255, 255))

player_detector = YoloV5()
ball_detector = YoloV5(model_path="models/best.pt")

classifier = NNClassifier(
    model_path="models/model_classification.pt",
    classes=["Chelsea", "Man City", "Referee"],
)


man_city = Team(name="Man City", abbreviation="MNC", color=(235, 206, 135))
chelsea = Team(name="Chelsea", abbreviation="CHE", color=(255, 0, 0))
teams = [man_city, chelsea]
match = Match(home=chelsea, away=man_city)
match.team_possession = man_city

motion_estimator = MotionEstimator()
coord_transformations = None

for frame in video:

    # Get Detections
    players_detections = get_player_detections(player_detector, classifier, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

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

    # for i, ball in enumerate(ball_detections):
    #     ball_detections[i].data["label"] = ball.data["id"]

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball)

    # Draw
    # frame = Player.draw_players(players=players, frame=frame, confidence=False)
    # if ball:
    #     frame = ball.draw(frame)
    # frame = path_drawer.draw(
    #     frame, ball_track_objects, coord_transform=coord_transformations
    # )
    frame = match.draw(frame, debug=True)

    # Write video
    video.write(frame)


# for i, detection in enumerate(tracked_detections):
#     tracked_detections[i].data["label"] = detection.data["id"]
