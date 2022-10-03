from typing import Iterable, List

import numpy as np
import PIL

from soccer.ball import Ball
from soccer.draw import AbsolutePath, PathPoint
from soccer.player import Player
from soccer.team import Team


class Pass:
    def __init__(
        self, start_ball_bbox: np.ndarray, end_ball_bbox: np.ndarray, team: Team
    ) -> None:
        # Abs coordinates
        self.start_ball_bbox = start_ball_bbox
        self.end_ball_bbox = end_ball_bbox
        self.team = team
        self.draw_abs = AbsolutePath()

    def draw(
        self, img: PIL.Image.Image, coord_transformations: "CoordinatesTransformation"
    ) -> PIL.Image.Image:
        """Draw a pass

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        coord_transformations : CoordinatesTransformation
            coordinates transformation

        Returns
        -------
        PIL.Image.Image
            frame with the new pass
        """
        rel_point_start = PathPoint.from_abs_bbox(
            id=0,
            abs_point=self.start_ball_bbox,
            coord_transformations=coord_transformations,
            color=self.team.color,
        )
        rel_point_end = PathPoint.from_abs_bbox(
            id=1,
            abs_point=self.end_ball_bbox,
            coord_transformations=coord_transformations,
            color=self.team.color,
        )

        new_pass = [rel_point_start, rel_point_end]

        pass_filtered = self.draw_abs.filter_points_outside_frame(
            path=new_pass,
            width=img.size[0],
            height=img.size[0],
            margin=3000,
        )

        if len(pass_filtered) == 2:
            img = self.draw_abs.draw_arrow(
                img=img, points=pass_filtered, color=self.team.color, width=6, alpha=150
            )

        return img

    @staticmethod
    def draw_pass_list(
        img: PIL.Image.Image,
        passes: List["Pass"],
        coord_transformations: "CoordinatesTransformation",
    ) -> PIL.Image.Image:
        """Draw all the passes

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        passes : List[Pass]
            Passes list to draw
        coord_transformations : CoordinatesTransformation
            Coordinate transformation for the current frame

        Returns
        -------
        PIL.Image.Image
            Drawed frame
        """
        for pass_ in passes:
            img = pass_.draw(img=img, coord_transformations=coord_transformations)

        return img

    def get_relative_coordinates(
        self, coord_transformations: "CoordinatesTransformation"
    ) -> tuple:
        """
        Print the relative coordinates of a pass

        Parameters
        ----------
        coord_transformations : CoordinatesTransformation
            Coordinates transformation

        Returns
        -------
        tuple
            (start, end) of the pass with relative coordinates
        """
        relative_start = coord_transformations.abs_to_rel(self.start_ball_bbox)
        relative_end = coord_transformations.abs_to_rel(self.end_ball_bbox)

        return (relative_start, relative_end)

    def get_center(self, points: np.array) -> tuple:
        """
        Returns the center of the points

        Parameters
        ----------
        points : np.array
            2D points

        Returns
        -------
        tuple
            (x, y) coordinates of the center
        """
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (center_x, center_y)

    def round_iterable(self, iterable: Iterable) -> Iterable:
        """
        Round all entries from one Iterable object

        Parameters
        ----------
        iterable : Iterable
            Iterable to round

        Returns
        -------
        Iterable
            Rounded Iterable
        """
        return [round(item) for item in iterable]

    def generate_output_pass(
        self, start: np.ndarray, end: np.ndarray, team_name: str
    ) -> str:
        """
        Generate a string with the pass information

        Parameters
        ----------
        start : np.ndarray
            The start point of the pass
        end : np.ndarray
            The end point of the pass
        team_name : str
            The team that did this pass

        Returns
        -------
        str
            String with the pass information
        """
        relative_start_point = self.get_center(start)
        relative_end_point = self.get_center(end)

        relative_start_round = self.round_iterable(relative_start_point)
        relative_end_round = self.round_iterable(relative_end_point)

        return f"Start: {relative_start_round}, End: {relative_end_round}, Team: {team_name}"

    def tostring(self, coord_transformations: "CoordinatesTransformation") -> str:
        """
        Get a string with the relative coordinates of this pass

        Parameters
        ----------
        coord_transformations : CoordinatesTransformation
            Coordinates transformation

        Returns
        -------
        str
            string with the relative coordinates
        """
        relative_start, relative_end = self.get_relative_coordinates(
            coord_transformations
        )

        return self.generate_output_pass(relative_start, relative_end, self.team.name)

    def __str__(self):
        return self.generate_output_pass(
            self.start_ball_bbox, self.end_ball_bbox, self.team.name
        )


class PassEvent:
    def __init__(self) -> None:
        self.ball = None
        self.closest_player = None
        self.init_player_with_ball = None
        self.last_player_with_ball = None
        self.player_with_ball_counter = 0
        self.player_with_ball_threshold = 3
        self.player_with_ball_threshold_dif_team = 4

    def update(self, closest_player: Player, ball: Ball) -> None:
        """
        Updates the player with the ball counter

        Parameters
        ----------
        closest_player : Player
            The closest player to the ball
        ball : Ball
            Ball class
        """
        self.ball = ball
        self.closest_player = closest_player

        same_id = Player.have_same_id(self.init_player_with_ball, closest_player)

        if same_id:
            self.player_with_ball_counter += 1
        elif not same_id:
            self.player_with_ball_counter = 0

        self.init_player_with_ball = closest_player

    def validate_pass(self, start_player: Player, end_player: Player) -> bool:
        """
        Check if there is a pass between two players of the same team

        Parameters
        ----------
        start_player : Player
            Player that originates the pass
        end_player : Player
            Destination player of the pass

        Returns
        -------
        bool
            Valid pass occurred
        """
        if Player.have_same_id(start_player, end_player):
            return False
        if start_player.team != end_player.team:
            return False

        return True

    def generate_pass(
        self, team: Team, start_pass: np.ndarray, end_pass: np.ndarray
    ) -> Pass:
        """
        Generate a new pass

        Parameters
        ----------
        team : Team
            Pass team
        start_pass : np.ndarray
            Pass start point
        end_pass : np.ndarray
            Pass end point

        Returns
        -------
        Pass
            The generated instance of the Pass class
        """
        start_pass_bbox = [start_pass, start_pass]

        new_pass = Pass(
            start_ball_bbox=start_pass_bbox,
            end_ball_bbox=end_pass,
            team=team,
        )

        return new_pass

    def process_pass(self) -> None:
        """
        Check if a new pass was generated and in the positive case save the new pass into de right team
        """
        if self.player_with_ball_counter >= self.player_with_ball_threshold:
            # init the last player with ball
            if self.last_player_with_ball is None:
                self.last_player_with_ball = self.init_player_with_ball

            valid_pass = self.validate_pass(
                start_player=self.last_player_with_ball,
                end_player=self.closest_player,
            )

            if valid_pass:
                # Generate new pass
                team = self.closest_player.team
                start_pass = self.last_player_with_ball.closest_foot_to_ball_abs(
                    self.ball
                )
                end_pass = self.ball.detection.absolute_points

                new_pass = self.generate_pass(
                    team=team, start_pass=start_pass, end_pass=end_pass
                )
                team.passes.append(new_pass)
            else:
                if (
                    self.player_with_ball_counter
                    < self.player_with_ball_threshold_dif_team
                ):
                    return

            self.last_player_with_ball = self.closest_player
