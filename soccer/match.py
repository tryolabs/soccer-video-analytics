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
        bar_height = 29
        bar_width = 310

        ratio = self.home.get_percentage_possession(self.duration)

        # Protect against too small rectangles
        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            origin,
            [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
        )

        right_rectangle = (
            [int(bar_x + ratio * bar_width), bar_y],
            [int(bar_x + bar_width), int(bar_y + bar_height)],
        )

        left_color = self.home.board_color
        right_color = self.away.board_color

        # Draw first one rectangle or another in orther to make the
        # rectangle bigger for better rounded corners
        if ratio < 0.15:
            left_rectangle[1][0] += 20

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=left_rectangle,
                color=left_color,
                radius=15,
            )

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=right_rectangle,
                color=right_color,
                left=True,
                radius=15,
            )
        else:
            right_rectangle[0][0] -= 20

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=right_rectangle,
                color=right_color,
                left=True,
                radius=15,
            )

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=left_rectangle,
                color=left_color,
                radius=15,
            )

        # Draw home text
        if ratio > 0.15:
            home_text = (
                f"{int(self.home.get_percentage_possession(self.duration) * 100)}%"
            )

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rectangle[0],
                width=left_rectangle[1][0] - left_rectangle[0][0],
                height=left_rectangle[1][1] - left_rectangle[0][1],
                text=home_text,
                color=self.home.text_color,
            )

        # Draw away text
        if ratio < 0.85:
            away_text = (
                f"{int(self.away.get_percentage_possession(self.duration) * 100)}%"
            )
            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rectangle[0],
                width=right_rectangle[1][0] - right_rectangle[0][0],
                height=right_rectangle[1][1] - right_rectangle[0][1],
                text=away_text,
                color=self.away.text_color,
            )

        return frame

    def get_counter_backround(
        self,
    ) -> PIL.Image.Image:
        """
        Get counter backround

        Returns
        -------
        PIL.Image.Image
            Counter backround
        """

        counter = PIL.Image.open("./images/board.png").convert("RGBA")
        counter = Draw.add_alpha(counter, 210)
        counter = np.array(counter)
        red, green, blue, alpha = counter.T
        counter = np.array([blue, green, red, alpha])
        counter = counter.transpose()
        counter = PIL.Image.fromarray(counter)
        counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
        return counter

    def draw_counter_background(
        self,
        frame: PIL.Image.Image,
        origin: tuple,
        counter_background: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Draw counter background

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)
        counter_background : PIL.Image.Image
            Counter background

        Returns
        -------
        PIL.Image.Image
            Frame with counter background
        """
        frame.paste(counter_background, origin, counter_background)
        return frame

    def draw_counter(
        self,
        frame: PIL.Image.Image,
        text: str,
        counter_text: str,
        origin: tuple,
        color: tuple,
        text_color: tuple,
        height: int = 27,
        width: int = 120,
    ) -> PIL.Image.Image:
        """
        Draw counter

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        text : str
            Text in left-side of counter
        counter_text : str
            Text in right-side of counter
        origin : tuple
            Origin (x, y)
        color : tuple
            Color
        text_color : tuple
            Color of text
        height : int, optional
            Height, by default 27
        width : int, optional
            Width, by default 120

        Returns
        -------
        PIL.Image.Image
            Frame with counter
        """

        team_begin = origin
        team_width_ratio = 0.417
        team_width = width * team_width_ratio

        team_rectangle = (
            team_begin,
            (team_begin[0] + team_width, team_begin[1] + height),
        )

        time_begin = (origin[0] + team_width, origin[1])
        time_width = width * (1 - team_width_ratio)

        time_rectangle = (
            time_begin,
            (time_begin[0] + time_width, time_begin[1] + height),
        )

        frame = Draw.half_rounded_rectangle(
            img=frame,
            rectangle=team_rectangle,
            color=color,
            radius=20,
        )

        frame = Draw.half_rounded_rectangle(
            img=frame,
            rectangle=time_rectangle,
            color=(239, 234, 229),
            radius=20,
            left=True,
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame,
            origin=team_rectangle[0],
            height=height,
            width=team_width,
            text=text,
            color=text_color,
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame,
            origin=time_rectangle[0],
            height=height,
            width=time_width,
            text=counter_text,
            color="black",
        )

        return frame

    def draw(
        self,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
    ) -> PIL.Image.Image:
        """

        Draw elements of the match in frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        counter_background : PIL.Image.Image
            Counter background
        debug : bool, optional
            Whether to draw extra debug information, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with elements of the match
        """

        # get width of PIL.Image
        frame_width = frame.size[0]
        counter_origin = (frame_width - 540, 40)

        frame = self.draw_counter_background(
            frame,
            origin=counter_origin,
            counter_background=counter_background,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, counter_origin[1] + 130),
            text=self.home.abbreviation,
            counter_text=self.home.get_time_possession(self.fps),
            color=self.home.board_color,
            text_color=self.home.text_color,
            height=31,
            width=150,
        )
        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 130),
            text=self.away.abbreviation,
            counter_text=self.away.get_time_possession(self.fps),
            color=self.away.board_color,
            text_color=self.away.text_color,
            height=31,
            width=150,
        )
        frame = self.possession_bar(
            frame, origin=(counter_origin[0] + 35, counter_origin[1] + 195)
        )

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
