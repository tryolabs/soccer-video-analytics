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

    def possession_bar(self, frame: np.ndarray) -> np.ndarray:
        """

        Draw possession bar

        Parameters
        ----------
        frame : np.ndarray
            Frame

        Returns
        -------
        np.ndarray
            Frame with possession bar
        """

        bar_x = 1920 - 507
        bar_y = 195
        bar_height = 30
        bar_width = 367

        ratio = self.home.get_percentage_possession(self.duration)

        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            [bar_x, bar_y],
            [int(bar_x + ratio * bar_width), bar_y + bar_height],
        )

        left_color = self.home.color

        new_frame = Draw.half_rounded_rectangle(
            frame,
            rectangle=left_rectangle,
            color=left_color,
            # left=True,
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
            font_size=0.8,
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
            font_size=0.8,
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
            font_size=0.8,
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
            font_size=0.8,
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
        tryo_logo = tryo_logo.resize((70, 70))
        tryo_logo = tryo_logo.convert("L")
        pil_frame = PIL.Image.fromarray(frame)
        pil_frame.paste(tryo_logo, origin, tryo_logo)
        return np.array(pil_frame)

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

        home_counter_origin = (1920 - 500, 140)
        new_frame = self.draw_home_counter(frame, origin=home_counter_origin)
        away_counter_origin = (1920 - 300, 140)
        new_frame = self.draw_away_counter(new_frame, origin=away_counter_origin)

        new_frame = self.add_tryolabs_logo(new_frame, origin=(1920 - 362, 55))

        if self.closest_player:
            new_frame = self.closest_player.draw_pointer(new_frame)

        new_frame = self.possession_bar(new_frame)

        if debug:
            # Draw line from closest player feet to ball
            if self.closest_player and self.ball:
                closest_foot = self.closest_player.closest_foot_to_ball(self.ball)

                color = (0, 0, 0)
                # Change line color if its greater than threshold
                distance = self.closest_player.distance_to_ball(self.ball)
                if distance > self.ball_distance_threshold:
                    color = (255, 255, 255)

                new_frame = cv2.line(
                    new_frame,
                    closest_foot,
                    self.ball.center,
                    color,
                    2,
                )

        return new_frame
