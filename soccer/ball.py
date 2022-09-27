import cv2
import norfair
import numpy as np

from soccer.draw import Draw


class Ball:
    def __init__(self, detection: norfair.Detection):
        """
        Initialize Ball

        Parameters
        ----------
        detection : norfair.Detection
            norfair.Detection containing the ball
        """
        self.detection = detection
        self.color = None

    def set_color(self, match: "Match"):
        """
        Sets the color of the ball to the team color with the ball possession in the match.

        Parameters
        ----------
        match : Match
            Match object
        """
        if match.team_possession is None:
            return

        self.color = match.team_possession.color

        if self.detection:
            self.detection.data["color"] = match.team_possession.color

    @property
    def center(self) -> tuple:
        """
        Returns the center of the ball

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            return None

        points = self.detection.points
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_y = (y1 + y2) / 2
        center_x = (x1 + x2) / 2

        return np.array([round(center_x), round(center_y)])

    @property
    def center_abs(self) -> tuple:
        """
        Returns the center of the ball

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            return None

        points = self.detection.absolute_points
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_y = (y1 + y2) / 2
        center_x = (x1 + x2) / 2

        return np.array([round(center_x), round(center_y)])

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the ball on the frame

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on

        Returns
        -------
        np.ndarray
            Frame with ball drawn
        """
        if self.detection is None:
            return frame

        return Draw.draw_detection(self.detection, frame)

    def __str__(self):
        return f"Ball: {self.center}"
