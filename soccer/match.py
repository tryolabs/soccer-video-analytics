from typing import List

import cv2
import numpy as np
import PIL

from soccer.ball import Ball
from soccer.draw import Draw
from soccer.player import Player
from soccer.team import Team


class Match:
    def __init__(self, home: Team, away: Team, fps: int = 30):
        """

        Initialize Match

        Parameters
        ----------
        home : Team
            Home team
        away : Team
            Away team
        fps : int, optional
            Fps, by default 30
        """
        self.duration = 0
        self.home = home
        self.away = away
        self.team_possession = self.home
        self.current_team = self.home
        self.possession_counter = 0
        self.possesion_counter_threshold = 20
        self.ball_distance_threshold = 60
        self.fps = fps

        self.closest_player = None
        self.ball = None

    def update(self, players: List[Player], ball: Ball):
        """

        Update match possession and closest player

        Parameters
        ----------
        players : List[Player]
            List of players
        ball : Ball
            Ball
        """

        self.update_possession()

        if ball is None or ball.detection is None:
            self.closest_player = None
            return

        self.ball = ball

        closest_player = min(players, key=lambda player: player.distance_to_ball(ball))

        self.closest_player = closest_player

        if closest_player.distance_to_ball(ball) > self.ball_distance_threshold:
            self.closest_player = None
            return

        # Reset counter if team changed
        if closest_player.team != self.current_team:
            self.possession_counter = 0
            self.current_team = closest_player.team

        self.possession_counter += 1

        if self.possession_counter >= self.possesion_counter_threshold:
            self.change_team(self.current_team)

    def change_team(self, team: Team):
        """

        Change team possession

        Parameters
        ----------
        team : Team, optional
            New team in possession
        """
        self.team_possession = team

    def update_possession(self):
        """
        Updates match duration and possession counter of team in possession
        """
        if self.team_possession is None:
            return

        self.team_possession.possession += 1
        self.duration += 1

    @property
    def home_possession_str(self) -> str:
        return f"{self.home.abbreviation}: {self.home.get_time_possession(self.fps)}"

    @property
    def away_possession_str(self) -> str:
        return f"{self.away.abbreviation}: {self.away.get_time_possession(self.fps)}"

    def __str__(self) -> str:
        return f"{self.home_possession_str} | {self.away_possession_str}"

    @property
    def time_possessions(self) -> str:
        return f"{self.home.name}: {self.home.get_time_possession(self.fps)} | {self.away.name}: {self.away.get_time_possession(self.fps)}"

    def possession_bar(self, frame: np.ndarray, origin: tuple) -> np.ndarray:
        """

        Draw possession bar

        Parameters
        ----------
        frame : np.ndarray
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        np.ndarray
            Frame with possession bar
        """

        bar_x = origin[0]
        bar_y = origin[1]
        bar_height = 30
        bar_width = 367

        ratio = self.home.get_percentage_possession(self.duration)

        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            origin,
            [int(bar_x + ratio * bar_width), bar_y + bar_height],
        )

        left_color = self.home.color

        new_frame = Draw.half_rounded_rectangle(
            frame,
            rectangle=left_rectangle,
            color=left_color,
        )

        right_rectangle = (
            [int(bar_x + ratio * bar_width), bar_y],
            [bar_x + bar_width, bar_y + bar_height],
        )

        right_color = self.away.color

        new_frame = Draw.half_rounded_rectangle(
            new_frame,
            rectangle=right_rectangle,
            color=right_color,
            left=True,
        )

        # Draw home text
        if ratio > 0.15:
            home_text = (
                f"{int(self.home.get_percentage_possession(self.duration) * 100)}%"
            )

            new_frame = Draw.text_in_middle_rectangle(
                img=new_frame,
                rectangle=left_rectangle,
                text=home_text,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.8,
                color=(255, 255, 255),
                thickness=2,
            )

        # Draw away text
        if ratio < 0.85:
            away_text = (
                f"{int(self.away.get_percentage_possession(self.duration) * 100)}%"
            )
            new_frame = Draw.text_in_middle_rectangle(
                img=new_frame,
                rectangle=right_rectangle,
                text=away_text,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.8,
                color=(255, 255, 255),
                thickness=2,
            )

        return new_frame

    def draw_home_counter(self, frame: np.ndarray, origin: tuple) -> np.ndarray:
        """
        Draw home counter

        Parameters
        ----------
        frame : np.ndarray
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        np.ndarray
            Frame with home counter
        """

        # Draw rectangle with opencv with starting point and height and width
        home_white_rectangle_begin = origin
        white_rectangle_width = 70
        height = 35

        # White rectangle for team abbreviation
        frame = Draw.draw_rectangle(
            img=frame,
            origin=home_white_rectangle_begin,
            width=white_rectangle_width,
            height=height,
            color=(255, 255, 255),
        )

        # Color rectangle
        color_rectangle_width = 7
        home_color_rectangle_begin = (
            home_white_rectangle_begin[0] - color_rectangle_width,
            home_white_rectangle_begin[1],
        )
        frame = Draw.draw_rectangle(
            img=frame,
            origin=home_color_rectangle_begin,
            width=color_rectangle_width,
            height=height,
            color=self.home.color,
        )

        # Red rectangle for time
        red_rectangle_width = 85
        home_red_rectangle_begin = (
            home_white_rectangle_begin[0] + white_rectangle_width,
            home_white_rectangle_begin[1],
        )
        frame = Draw.draw_rectangle(
            img=frame,
            origin=home_red_rectangle_begin,
            width=red_rectangle_width,
            height=height,
            color=(52, 66, 53),
        )

        # Draw text
        text_height = home_white_rectangle_begin[1] + 25
        home_abbreviation_text_begin = (
            home_white_rectangle_begin[0] + 5,
            text_height,
        )
        frame = Draw.draw_text(
            img=frame,
            origin=home_abbreviation_text_begin,
            text=self.home.abbreviation,
            font_scale=0.8,
            color=(85, 80, 82),
        )

        home_time_text_begin = (
            home_red_rectangle_begin[0] + 5,
            text_height,
        )
        frame = Draw.draw_text(
            img=frame,
            origin=home_time_text_begin,
            text=self.home.get_time_possession(self.fps),
            font_scale=0.8,
            color=(255, 255, 255),
        )

        return frame

    def draw_away_counter(self, frame: np.ndarray, origin: tuple) -> np.ndarray:
        """
        Draw away counter

        Parameters
        ----------
        frame : np.ndarray
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        np.ndarray
            Frame with away counter
        """

        # Draw rectangle with opencv with starting point and height and width
        away_white_rectangle_begin = origin
        height = 35

        # Red rectangle for time
        red_rectangle_width = 85
        away_red_rectangle_begin = (
            away_white_rectangle_begin[0],
            away_white_rectangle_begin[1],
        )

        frame = Draw.draw_rectangle(
            img=frame,
            origin=away_red_rectangle_begin,
            width=red_rectangle_width,
            height=height,
            color=(52, 66, 53),
        )

        white_rectangle_width = 70
        away_white_rectangle_begin = (
            away_red_rectangle_begin[0] + red_rectangle_width,
            away_red_rectangle_begin[1],
        )
        # White rectangle for team abbreviation
        frame = Draw.draw_rectangle(
            img=frame,
            origin=away_white_rectangle_begin,
            width=white_rectangle_width,
            height=height,
            color=(255, 255, 255),
        )

        # Color rectangle
        color_rectangle_width = 7
        away_color_rectangle_begin = (
            away_white_rectangle_begin[0] + white_rectangle_width,
            away_white_rectangle_begin[1],
        )
        frame = Draw.draw_rectangle(
            img=frame,
            origin=away_color_rectangle_begin,
            width=color_rectangle_width,
            height=height,
            color=self.away.color,
        )

        # Draw text
        text_height = away_white_rectangle_begin[1] + 25
        away_abbreviation_text_begin = (
            away_white_rectangle_begin[0] + 5,
            text_height,
        )
        frame = Draw.draw_text(
            img=frame,
            origin=away_abbreviation_text_begin,
            text=self.away.abbreviation,
            font_scale=0.8,
            color=(85, 80, 82),
        )

        away_time_text_begin = (
            away_red_rectangle_begin[0] + 5,
            text_height,
        )
        frame = Draw.draw_text(
            img=frame,
            origin=away_time_text_begin,
            text=self.away.get_time_possession(self.fps),
            font_scale=0.8,
            color=(255, 255, 255),
        )

        return frame

    def add_tryolabs_logo(self, frame: np.ndarray, origin: tuple) -> np.ndarray:
        """
        Inserts tryolabs logo into image with PNG transparency

        Parameters
        ----------
        frame : np.ndarray
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        np.ndarray
            Frame with tryolabs logo
        """

        tryo_logo = PIL.Image.open("./images/tryo_logo.png").convert("RGBA")

        # Convert RGBA to BGRA
        tryo_logo = np.array(tryo_logo)
        red, green, blue, alpha = tryo_logo.T
        tryo_logo = np.array([blue, green, red, alpha])
        tryo_logo = tryo_logo.transpose()
        tryo_logo = PIL.Image.fromarray(tryo_logo)

        tryo_logo = tryo_logo.resize((70, 70))
        pil_frame = PIL.Image.fromarray(frame)
        pil_frame.paste(tryo_logo, origin, tryo_logo)
        return np.array(pil_frame)

    def draw_title(self, frame: np.ndarray, origin: tuple) -> np.ndarray:
        """
        Draw title

        Parameters
        ----------
        frame : np.ndarray
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        np.ndarray
            Frame with title
        """

        width = 370
        height = 30

        rectangle = (origin, (origin[0] + width, origin[1] + height))

        frame = Draw.rounded_rectangle(
            img=frame,
            rectangle=rectangle,
            radius=20,
            color=(255, 255, 255),
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame,
            rectangle=rectangle,
            text="Ball Possession",
            font_scale=0.8,
            color=(85, 80, 82),
        )
        return frame

    def draw(self, frame: np.ndarray, debug: bool = False) -> np.ndarray:
        """

        Draw elements of the match in frame

        Parameters
        ----------
        frame : np.ndarray
            Frame
        debug : bool, optional
            Whether to draw extra debug information, by default False

        Returns
        -------
        np.ndarray
            Frame with elements of the match
        """
        frame_width = frame.shape[1]
        frame = self.draw_title(frame=frame, origin=(frame_width - 500 - 7, 90))
        frame = self.draw_home_counter(frame, origin=(frame_width - 500, 140))
        frame = self.draw_away_counter(frame, origin=(frame_width - 300, 140))
        frame = self.add_tryolabs_logo(frame, origin=(frame_width - 362, 12))
        frame = self.possession_bar(frame, origin=(frame_width - 507, 195))

        if self.closest_player:
            frame = self.closest_player.draw_pointer(frame)

        if debug:
            # Draw line from closest player feet to ball
            if self.closest_player and self.ball:
                closest_foot = self.closest_player.closest_foot_to_ball(self.ball)

                color = (0, 0, 0)
                # Change line color if its greater than threshold
                distance = self.closest_player.distance_to_ball(self.ball)
                if distance > self.ball_distance_threshold:
                    color = (255, 255, 255)

                frame = cv2.line(
                    frame,
                    closest_foot,
                    self.ball.center,
                    color,
                    2,
                )

        return frame
