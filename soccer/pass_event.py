from typing import List

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

        init_player = self.init_player_with_ball

        init_player_has_id = init_player and "id" in init_player.detection.data
        closest_player_has_id = closest_player and "id" in closest_player.detection.data

        same_id = (
            init_player_has_id
            and closest_player_has_id
            and (init_player == closest_player)
        )

        different_id = (
            init_player_has_id
            and closest_player_has_id
            and not (init_player == closest_player)
        )

        if same_id:
            self.player_with_ball_counter += 1
        elif different_id:
            self.player_with_ball_counter = 0

        self.init_player_with_ball = closest_player

    def validate_pass(self, posible_start: Player, posible_end: Player) -> bool:
        """
        Check if there is a pass between two players of the same team

        Parameters
        ----------
        posible_start : Player
            Player that originates the pass
        posible_end : Player
            Destination player of the pass

        Returns
        -------
        bool
            Valid pass occurred
        """
        last_player_has_id = posible_start and "id" in posible_start.detection.data
        closest_player_has_id = posible_end and "id" in posible_end.detection.data
        players_id = last_player_has_id and closest_player_has_id

        different_player = players_id and not (posible_start == posible_end)
        same_team = posible_start.team == posible_end.team

        return different_player and same_team

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
                posible_start=self.last_player_with_ball,
                posible_end=self.closest_player,
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
