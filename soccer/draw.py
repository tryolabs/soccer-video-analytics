from typing import Union

import cv2
import norfair
import numpy as np
import pandas as pd
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
        font_size: float = 0.5,
        color: tuple = (255, 255, 255),
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size
        line_type = 2

        return cv2.putText(
            img, text, origin, font, font_scale, color, line_type, cv2.LINE_AA
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
