import random
from typing import List, Union

import cv2
import norfair
import numpy as np
import PIL


class Draw:
    @staticmethod
    def draw_rectangle(
        img: np.ndarray,
        origin: tuple,
        width: int,
        height: int,
        color: tuple,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw a rectangle on the image

        Parameters
        ----------
        img : np.ndarray
            Image
        origin : tuple
            Origin of the rectangle (x, y)
        width : int
            Width of the rectangle
        height : int
            Height of the rectangle
        color : tuple
            Color of the rectangle (BGR)
        thickness : int, optional
            Thickness of the rectangle, by default 2

        Returns
        -------
        np.ndarray
            Image with the rectangle drawn
        """
        return cv2.rectangle(
            img, origin, (origin[0] + width, origin[1] + height), color, -1
        )

    @staticmethod
    def draw_text(
        img: np.ndarray,
        origin: tuple,
        text: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        color: tuple = (255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw text on the image

        Parameters
        ----------
        img : np.ndarray
            Image
        origin : tuple
            Origin of the text (x, y)
        text : str
            Text to draw
        font_size : float, optional
            Font Size, by default 0.5
        color : tuple, optional
            Color, by default (255, 255, 255)

        Returns
        -------
        np.ndarray
            Image with the text drawn
        """

        return cv2.putText(
            img, text, origin, font, font_scale, color, thickness, cv2.LINE_AA
        )

    @staticmethod
    def draw_bounding_box(
        img: np.ndarray, rectangle: tuple, color: tuple, thickness: int = 2
    ) -> np.ndarray:
        """

        Draw a bounding box on the image

        Parameters
        ----------
        img : np.ndarray
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        thickness : int, optional
            Thickness of the rectangle, by default 2

        Returns
        -------
        np.ndarray
            Image with the bounding box drawn
        """
        return cv2.rectangle(img, rectangle[0], rectangle[1], color, thickness)

    @staticmethod
    def draw_detection(
        detection: norfair.Detection, img: np.ndarray, condifence: bool = False
    ) -> np.ndarray:
        """
        Draw a bounding box on the image from a norfair.Detection

        Parameters
        ----------
        detection : norfair.Detection
            Detection to draw
        img : np.ndarray
            Image
        condifence : bool, optional
            Whether to draw confidence in the box, by default False

        Returns
        -------
        np.ndarray
            Image with the bounding box drawn
        """

        if detection is None:
            return img

        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]

        color = (0, 0, 0)
        if "color" in detection.data:
            color = detection.data["color"]

        Draw.draw_bounding_box(img=img, rectangle=detection.points, color=color)

        if "label" in detection.data:
            label = detection.data["label"]
            img = cv2.putText(
                img,
                str(label),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        if condifence:
            img = cv2.putText(
                img,
                str(round(detection.data["p"], 2)),
                (x2, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return img

    @staticmethod
    def draw_pointer(
        detection: norfair.Detection, img: np.ndarray, color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """

        Draw a pointer on the image from a norfair.Detection bounding box

        Parameters
        ----------
        detection : norfair.Detection
            Detection to draw
        img : np.ndarray
            Image
        color : tuple, optional
            Pointer color, by default (0, 255, 0)

        Returns
        -------
        np.ndarray
            Image with the pointer drawn
        """
        if detection is None:
            return

        if color is None:
            color = (0, 0, 0)

        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]

        img = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(img)

        # (t_x1, t_y1)        (t_x2, t_y2)
        #   \                  /
        #    \                /
        #     \              /
        #      \            /
        #       \          /
        #        \        /
        #         \      /
        #          \    /
        #           \  /
        #       (t_x3, t_y3)

        width = 20
        height = 20
        vertical_space_from_bbox = 7

        t_x3 = 0.5 * x1 + 0.5 * x2
        t_x1 = t_x3 - width / 2
        t_x2 = t_x3 + width / 2

        t_y1 = y1 - vertical_space_from_bbox - height
        t_y2 = t_y1
        t_y3 = y1 - vertical_space_from_bbox

        draw.polygon(
            [
                (t_x1, t_y1),
                (t_x2, t_y2),
                (t_x3, t_y3),
            ],
            fill=color,
        )

        draw.line(
            [
                (t_x1, t_y1),
                (t_x2, t_y2),
                (t_x3, t_y3),
                (t_x1, t_y1),
            ],
            fill="black",
            width=2,
        )

        return np.array(img)

    @staticmethod
    def rounded_rectangle(
        img: np.ndarray, rectangle: tuple, color: tuple, radius: int = 15
    ) -> np.ndarray:
        """
        Draw a rounded rectangle on the image

        Parameters
        ----------
        img : np.ndarray
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        radius : int, optional
            Radius of the corners, by default 15

        Returns
        -------
        np.ndarray
            Image with the rounded rectangle drawn
        """

        overlay = img.copy()
        overlay = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle(rectangle, radius, fill=color)
        return np.array(overlay)

    @staticmethod
    def half_rounded_rectangle(
        img: np.ndarray,
        rectangle: tuple,
        color: tuple,
        radius: int = 15,
        left: bool = False,
    ) -> np.ndarray:
        """

        Draw a half rounded rectangle on the image

        Parameters
        ----------
        img : np.ndarray
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        radius : int, optional
            Radius of the rounded borders, by default 15
        left : bool, optional
            Whether the flat side is the left side, by default False

        Returns
        -------
        np.ndarray
            Image with the half rounded rectangle drawn
        """
        overlay = img.copy()

        overlay = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle(rectangle, radius, fill=color)

        height = rectangle[1][1] - rectangle[0][1]
        stop_width = 15

        if left:
            draw.rectangle(
                (
                    rectangle[0][0] + 0,
                    rectangle[1][1] - height,
                    rectangle[0][0] + stop_width,
                    rectangle[1][1],
                ),
                fill=color,
            )
        else:
            draw.rectangle(
                (
                    rectangle[1][0] - stop_width,
                    rectangle[1][1] - height,
                    rectangle[1][0],
                    rectangle[1][1],
                ),
                fill=color,
            )
        return np.array(overlay)

    @staticmethod
    def text_in_middle_rectangle(
        img: np.ndarray,
        rectangle: tuple,
        text: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.8,
        color=(255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        half_rectangle = [
            (rectangle[0][0] + rectangle[1][0]) / 2,
            (rectangle[0][1] + rectangle[1][1]) / 2,
        ]

        text_x = int(half_rectangle[0] - text_size[0] / 2)
        text_y = int(half_rectangle[1] + text_size[1] / 2)

        img = Draw.draw_text(
            img=img,
            text=text,
            origin=(text_x, text_y),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.8,
            color=color,
            thickness=2,
        )

        return img


class BallPoint:
    def __init__(
        self, id: int, point: tuple, color: tuple = (255, 255, 255), alpha: float = 1
    ):
        self.id = id
        self.point = point
        self.color = color
        self.alpha = alpha

    def __str__(self) -> str:
        return str(self.id)


class AbsolutePath:
    def __init__(self) -> None:
        self.past_points = []

    def center(self, points: np.ndarray) -> tuple:
        return (
            int((points[0][0] + points[1][0]) / 2),
            int((points[0][1] + points[1][1]) / 2),
        )

    def draw_path_with_pil_slow(
        self,
        img: np.ndarray,
        path: List[BallPoint],
        color: tuple = (255, 255, 255),
        thickness: int = 4,
    ) -> np.ndarray:
        """
        Draw a path with PIL

        Parameters
        ----------
        img : np.ndarray
            Image
        path : list
            List of points
        color : tuple, optional
            Color of the path, by default (255, 255, 255)
        thickness : int, optional
            Thickness of the path, by default 2

        Returns
        -------
        np.ndarray
            Image with the path drawn
        """
        img = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(img, "RGBA")
        for i in range(len(path) - 1):
            color_with_alpha = tuple(
                [color[0], color[1], color[2], int(path[i].alpha * 255)]
            )
            draw.line(
                [path[i].point, path[i + 1].point],
                fill=color_with_alpha,
                width=thickness,
            )
        return np.array(img)

    def draw_path_with_pil_fast(
        self, img: np.ndarray, path: List[tuple], color: tuple
    ) -> np.ndarray:

        img = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(img)

        draw.line(
            path,
            fill=color,
            width=2,
        )

        return np.array(img)

    def draw_path_with_cv2_fast(
        self, img: np.ndarray, path: List[tuple], color: tuple
    ) -> np.ndarray:
        return cv2.polylines(img, [np.array(path)], False, color, 2)

    def draw_path_with_cv2_slow(
        self, img: np.ndarray, ball_path: List[BallPoint], color: tuple
    ) -> np.ndarray:

        # return img

        overlay = img.copy()

        for j, ball_point in enumerate(ball_path):

            if j > 0:
                previous_point = ball_path[j - 1].point
                point = ball_point.point

                cv2.line(
                    overlay,
                    previous_point,
                    point,
                    color=color,
                    thickness=2,
                )

            img = cv2.addWeighted(
                overlay, ball_point.alpha, img, 1 - ball_point.alpha, 0
            )

        return img

    def add_new_point(self, detection: norfair.Detection) -> None:

        if detection is None:
            return

        self.past_points.insert(0, detection.absolute_points)

    def draw(
        self,
        img: np.ndarray,
        detection: norfair.Detection,
        coord_transformations,
        alpha: float = 0.5,
        color: tuple = (255, 255, 255),
    ) -> np.ndarray:

        self.add_new_point(detection=detection)

        if len(self.past_points) < 2:
            return img

        path = [
            self.center(coord_transformations.abs_to_rel(past_point))
            for past_point in self.past_points
        ]

        # random between 0 and 1
        # alpha = random.random()
        # (len(path) - i) / len(path))
        ball_path = []
        # alpha = 1
        for i, point in enumerate(path):
            alpha = (len(path) - i) / len(path)
            ball_path.append(BallPoint(id=i, point=point, alpha=alpha))
            # alpha *= 0.994

        # ball_path = [
        #     BallPoint(id=i, point=point, alpha=)
        #     for i, point in enumerate(path)
        # ]

        margin = 150
        ball_path = [
            ball_point
            for ball_point in ball_path
            if ball_point.point[0] > 0 - margin
            and ball_point.point[1] > 0 - margin
            and ball_point.point[0] < img.shape[1] + margin
            and ball_point.point[1] < img.shape[0] + margin
        ]

        # return self.draw_path_with_pil_fast(img, path, color=(255, 255, 255, 120))
        # return self.draw_path_with_cv2_slow(img, ball_path, color=(255, 255, 255))
        return self.draw_path_with_pil_slow(img, ball_path, color=(255, 255, 255))
