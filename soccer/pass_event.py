from typing import List

import numpy as np
import PIL

from soccer.draw import AbsolutePath, PathPoint


class Pass:
    def __init__(
        self, start_ball_bbox: np.ndarray, end_ball_bbox: np.ndarray, team: "Team"
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

    def update(self, closest_player: "Player", ball: "Ball") -> None:
        """
        Updates the player with the ball counter
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

    def process_pass(self) -> None:
        """
        Check if a new pass was generated
        """
        if self.player_with_ball_counter >= self.player_with_ball_threshold:
            # init the last player with ball
            if self.last_player_with_ball is None:
                self.last_player_with_ball = self.init_player_with_ball

            last_player_has_id = (
                self.last_player_with_ball
                and "id" in self.last_player_with_ball.detection.data
            )
            closest_player_has_id = (
                self.closest_player and "id" in self.closest_player.detection.data
            )
            players_id = last_player_has_id and closest_player_has_id

            different_player = players_id and not (
                self.last_player_with_ball == self.closest_player
            )
            same_team = self.last_player_with_ball.team == self.closest_player.team

            if different_player and same_team:
                # Generate new pass
                start_pass = self.last_player_with_ball.closest_foot_to_ball_abs(
                    self.ball
                )
                start_pass_bbox = [start_pass, start_pass]

                team = self.closest_player.team

                new_pass = Pass(
                    start_ball_bbox=start_pass_bbox,
                    end_ball_bbox=self.ball.detection.absolute_points,
                    team=team,
                )
                team.passes.append(new_pass)
            else:
                if (
                    self.player_with_ball_counter
                    < self.player_with_ball_threshold_dif_team
                ):
                    return

            self.last_player_with_ball = self.closest_player
