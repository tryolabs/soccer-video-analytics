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

    def draw(self, img: PIL.Image.Image, coord_transformations) -> PIL.Image.Image:
        """Draw a pass

        Args:
            img (PIL.Image.Image): frame
            coord_transformations: coordinates transformation

        Returns:
            PIL.Image.Image: frame with the new pass
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
            margin=2500,
        )

        if len(pass_filtered) == 2:
            img = self.draw_abs.draw_arrow(
                img=img, points=pass_filtered, color=self.team.color, width=6
            )

        return img

    @staticmethod
    def draw_pass_list(
        img: PIL.Image.Image, passes: List["Pass"], coord_transformations
    ) -> PIL.Image.Image:
        for pass_ in passes:
            img = pass_.draw(img=img, coord_transformations=coord_transformations)

        return img
