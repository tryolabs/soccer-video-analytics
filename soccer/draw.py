from math import sqrt
from typing import List

import norfair
import numpy as np
import PIL


class Draw:
    @staticmethod
    def draw_rectangle(
        img: PIL.Image.Image,
        origin: tuple,
        width: int,
        height: int,
        color: tuple,
        thickness: int = 2,
    ) -> PIL.Image.Image:
        """
        Draw a rectangle on the image

        Parameters
        ----------
        img : PIL.Image.Image
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
        PIL.Image.Image
            Image with the rectangle drawn
        """

        draw = PIL.ImageDraw.Draw(img)
        draw.rectangle(
            [origin, (origin[0] + width, origin[1] + height)],
            fill=color,
            width=thickness,
        )
        return img

    @staticmethod
    def draw_text(
        img: PIL.Image.Image,
        origin: tuple,
        text: str,
        font: PIL.ImageFont = None,
        color: tuple = (255, 255, 255),
    ) -> PIL.Image.Image:
        """
        Draw text on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        origin : tuple
            Origin of the text (x, y)
        text : str
            Text to draw
        font : PIL.ImageFont
            Font to use
        color : tuple, optional
            Color of the text (RGB), by default (255, 255, 255)

        Returns
        -------
        PIL.Image.Image
        """
        draw = PIL.ImageDraw.Draw(img)

        if font is None:
            font = PIL.ImageFont.truetype("fonts/Gidole-Regular.ttf", size=20)

        draw.text(
            origin,
            text,
            font=font,
            fill=color,
        )

        return img

    @staticmethod
    def draw_bounding_box(
        img: PIL.Image.Image, rectangle: tuple, color: tuple, thickness: int = 3
    ) -> PIL.Image.Image:
        """

        Draw a bounding box on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        thickness : int, optional
            Thickness of the rectangle, by default 2

        Returns
        -------
        PIL.Image.Image
            Image with the bounding box drawn
        """

        rectangle = rectangle[0:2]

        draw = PIL.ImageDraw.Draw(img)
        rectangle = [tuple(x) for x in rectangle]
        # draw.rectangle(rectangle, outline=color, width=thickness)
        draw.rounded_rectangle(rectangle, radius=7, outline=color, width=thickness)

        return img

    @staticmethod
    def draw_detection(
        detection: norfair.Detection,
        img: PIL.Image.Image,
        confidence: bool = False,
        id: bool = False,
    ) -> PIL.Image.Image:
        """
        Draw a bounding box on the image from a norfair.Detection

        Parameters
        ----------
        detection : norfair.Detection
            Detection to draw
        img : PIL.Image.Image
            Image
        confidence : bool, optional
            Whether to draw confidence in the box, by default False
        id : bool, optional
            Whether to draw id in the box, by default False

        Returns
        -------
        PIL.Image.Image
            Image with the bounding box drawn
        """

        if detection is None:
            return img

        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]

        color = (0, 0, 0)
        if "color" in detection.data:
            color = detection.data["color"] + (255,)

        img = Draw.draw_bounding_box(img=img, rectangle=detection.points, color=color)

        if "label" in detection.data:
            label = detection.data["label"]
            img = Draw.draw_text(
                img=img,
                origin=(x1, y1 - 20),
                text=label,
                color=color,
            )

        if "id" in detection.data and id is True:
            id = detection.data["id"]
            img = Draw.draw_text(
                img=img,
                origin=(x2, y1 - 20),
                text=f"ID: {id}",
                color=color,
            )

        if confidence:
            img = Draw.draw_text(
                img=img,
                origin=(x1, y2),
                text=str(round(detection.data["p"], 2)),
                color=color,
            )

        return img

    @staticmethod
    def draw_pointer(
        detection: norfair.Detection, img: PIL.Image.Image, color: tuple = (0, 255, 0)
    ) -> PIL.Image.Image:
        """

        Draw a pointer on the image from a norfair.Detection bounding box

        Parameters
        ----------
        detection : norfair.Detection
            Detection to draw
        img : PIL.Image.Image
            Image
        color : tuple, optional
            Pointer color, by default (0, 255, 0)

        Returns
        -------
        PIL.Image.Image
            Image with the pointer drawn
        """
        if detection is None:
            return

        if color is None:
            color = (0, 0, 0)

        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]

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

        return img

    @staticmethod
    def rounded_rectangle(
        img: PIL.Image.Image, rectangle: tuple, color: tuple, radius: int = 15
    ) -> PIL.Image.Image:
        """
        Draw a rounded rectangle on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        radius : int, optional
            Radius of the corners, by default 15

        Returns
        -------
        PIL.Image.Image
            Image with the rounded rectangle drawn
        """

        overlay = img.copy()
        draw = PIL.ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle(rectangle, radius, fill=color)
        return overlay

    @staticmethod
    def half_rounded_rectangle(
        img: PIL.Image.Image,
        rectangle: tuple,
        color: tuple,
        radius: int = 15,
        left: bool = False,
    ) -> PIL.Image.Image:
        """

        Draw a half rounded rectangle on the image

        Parameters
        ----------
        img : PIL.Image.Image
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
        PIL.Image.Image
            Image with the half rounded rectangle drawn
        """
        overlay = img.copy()
        draw = PIL.ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle(rectangle, radius, fill=color)

        height = rectangle[1][1] - rectangle[0][1]
        stop_width = 13

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
        return overlay

    @staticmethod
    def text_in_middle_rectangle(
        img: PIL.Image.Image,
        origin: tuple,
        width: int,
        height: int,
        text: str,
        font: PIL.ImageFont = None,
        color=(255, 255, 255),
    ) -> PIL.Image.Image:
        """
        Draw text in middle of rectangle

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        origin : tuple
            Origin of the rectangle (x, y)
        width : int
            Width of the rectangle
        height : int
            Height of the rectangle
        text : str
            Text to draw
        font : PIL.ImageFont, optional
            Font to use, by default None
        color : tuple, optional
            Color of the text, by default (255, 255, 255)

        Returns
        -------
        PIL.Image.Image
            Image with the text drawn
        """

        draw = PIL.ImageDraw.Draw(img)

        if font is None:
            font = PIL.ImageFont.truetype("fonts/Gidole-Regular.ttf", size=24)

        w, h = draw.textsize(text, font=font)
        text_origin = (
            origin[0] + width / 2 - w / 2,
            origin[1] + height / 2 - h / 2,
        )

        draw.text(text_origin, text, font=font, fill=color)

        return img

    @staticmethod
    def add_alpha(img: PIL.Image.Image, alpha: int = 100) -> PIL.Image.Image:
        """
        Add an alpha channel to an image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        alpha : int, optional
            Alpha value, by default 100

        Returns
        -------
        PIL.Image.Image
            Image with alpha channel
        """
        data = img.getdata()
        newData = []
        for old_pixel in data:

            # Don't change transparency of transparent pixels
            if old_pixel[3] != 0:
                pixel_with_alpha = old_pixel[:3] + (alpha,)
                newData.append(pixel_with_alpha)
            else:
                newData.append(old_pixel)

        img.putdata(newData)
        return img


class PathPoint:
    def __init__(
        self, id: int, center: tuple, color: tuple = (255, 255, 255), alpha: float = 1
    ):
        """
        Path point

        Parameters
        ----------
        id : int
            Id of the point
        center : tuple
            Center of the point (x, y)
        color : tuple, optional
            Color of the point, by default (255, 255, 255)
        alpha : float, optional
            Alpha value of the point, by default 1
        """
        self.id = id
        self.center = center
        self.color = color
        self.alpha = alpha

    def __str__(self) -> str:
        return str(self.id)

    @property
    def color_with_alpha(self) -> tuple:
        return (self.color[0], self.color[1], self.color[2], int(self.alpha * 255))

    @staticmethod
    def get_center_from_bounding_box(bounding_box: np.ndarray) -> tuple:
        """
        Get the center of a bounding box

        Parameters
        ----------
        bounding_box : np.ndarray
            Bounding box [[xmin, ymin], [xmax, ymax]]

        Returns
        -------
        tuple
            Center of the bounding box (x, y)
        """
        return (
            int((bounding_box[0][0] + bounding_box[1][0]) / 2),
            int((bounding_box[0][1] + bounding_box[1][1]) / 2),
        )

    @staticmethod
    def from_abs_bbox(
        id: int,
        abs_point: np.ndarray,
        coord_transformations,
        color: tuple = None,
        alpha: float = None,
    ) -> "PathPoint":
        """
        Create a PathPoint from an absolute bounding box.
        It converts the absolute bounding box to a relative one and then to a center point

        Parameters
        ----------
        id : int
            Id of the point
        abs_point : np.ndarray
            Absolute bounding box
        coord_transformations : "CoordTransformations"
            Coordinate transformations
        color : tuple, optional
            Color of the point, by default None
        alpha : float, optional
            Alpha value of the point, by default None

        Returns
        -------
        PathPoint
            PathPoint
        """

        rel_point = coord_transformations.abs_to_rel(abs_point)
        center = PathPoint.get_center_from_bounding_box(rel_point)

        return PathPoint(id=id, center=center, color=color, alpha=alpha)


class AbsolutePath:
    def __init__(self) -> None:
        self.past_points = []
        self.color_by_index = {}

    def center(self, points: np.ndarray) -> tuple:
        """
        Get the center of a Norfair Bounding Box Detection point

        Parameters
        ----------
        points : np.ndarray
            Norfair Bounding Box Detection point

        Returns
        -------
        tuple
            Center of the point (x, y)
        """
        return (
            int((points[0][0] + points[1][0]) / 2),
            int((points[0][1] + points[1][1]) / 2),
        )

    @property
    def path_length(self) -> int:
        return len(self.past_points)

    def draw_path_slow(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        thickness: int = 4,
    ) -> PIL.Image.Image:
        """
        Draw a path with alpha

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            List of points to draw
        thickness : int, optional
            Thickness of the path, by default 4

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        for i in range(len(path) - 1):
            draw.line(
                [path[i].center, path[i + 1].center],
                fill=path[i].color_with_alpha,
                width=thickness,
            )
        return img

    def draw_arrow_head(
        self,
        img: PIL.Image.Image,
        start: tuple,
        end: tuple,
        color: tuple = (255, 255, 255),
        length: int = 10,
        height: int = 6,
        thickness: int = 4,
        alpha: int = 255,
    ) -> PIL.Image.Image:

        # https://stackoverflow.com/questions/43527894/drawing-arrowheads-which-follow-the-direction-of-the-line-in-pygame
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        dX = end[0] - start[0]
        dY = end[1] - start[1]

        # vector length
        Len = sqrt(dX * dX + dY * dY)  # use Hypot if available

        if Len == 0:
            return img

        # normalized direction vector components
        udX = dX / Len
        udY = dY / Len

        # perpendicular vector
        perpX = -udY
        perpY = udX

        # points forming arrowhead
        # with length L and half-width H
        arrowend = end

        leftX = end[0] - length * udX + height * perpX
        leftY = end[1] - length * udY + height * perpY

        rightX = end[0] - length * udX - height * perpX
        rightY = end[1] - length * udY - height * perpY

        if len(color) <= 3:
            color += (alpha,)

        draw.line(
            [(leftX, leftY), arrowend],
            fill=color,
            width=thickness,
        )

        draw.line(
            [(rightX, rightY), arrowend],
            fill=color,
            width=thickness,
        )

        return img

    def draw_path_arrows(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        thickness: int = 4,
        frame_frequency: int = 30,
    ) -> PIL.Image.Image:
        """
        Draw a path with arrows every 30 points

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            Path
        thickness : int, optional
            Thickness of the path, by default 4

        Returns
        -------
        PIL.Image.Image
            Image with the arrows drawn
        """

        for i, point in enumerate(path):

            if i < 4 or i % frame_frequency:
                continue

            end = path[i]
            start = path[i - 4]

            img = self.draw_arrow_head(
                img=img,
                start=start.center,
                end=end.center,
                color=start.color_with_alpha,
                thickness=thickness,
            )

        return img

    def draw_path_fast(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        color: tuple,
        width: int = 2,
        alpha: int = 255,
    ) -> PIL.Image.Image:
        """
        Draw a path without alpha (faster)

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            Path
        color : tuple
            Color of the path
        with : int
            Width of the line
        alpha : int
            Color alpha (0-255)

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        path_list = [point.center for point in path]

        color += (alpha,)

        draw.line(
            path_list,
            fill=color,
            width=width,
        )

        return img

    def draw_arrow(
        self,
        img: PIL.Image.Image,
        points: List[PathPoint],
        color: tuple,
        width: int,
        alpha: int = 255,
    ) -> PIL.Image.Image:
        """Draw arrow between two points

        Parameters
        ----------
        img : PIL.Image.Image
            image to draw
        points : List[PathPoint]
            start and end points
        color : tuple
            color of the arrow
        width : int
            width of the arrow
        alpha : int, optional
            color alpha (0-255), by default 255

        Returns
        -------
        PIL.Image.Image
            Image with the arrow
        """

        img = self.draw_path_fast(
            img=img, path=points, color=color, width=width, alpha=alpha
        )
        img = self.draw_arrow_head(
            img=img,
            start=points[0].center,
            end=points[1].center,
            color=color,
            length=30,
            height=15,
            alpha=alpha,
        )

        return img

    def add_new_point(
        self, detection: norfair.Detection, color: tuple = (255, 255, 255)
    ) -> None:
        """
        Add a new point to the path

        Parameters
        ----------
        detection : norfair.Detection
            Detection
        color : tuple, optional
            Color of the point, by default (255, 255, 255)
        """

        if detection is None:
            return

        self.past_points.append(detection.absolute_points)

        self.color_by_index[len(self.past_points) - 1] = color

    def filter_points_outside_frame(
        self, path: List[PathPoint], width: int, height: int, margin: int = 0
    ) -> List[PathPoint]:
        """
        Filter points outside the frame with a margin

        Parameters
        ----------
        path : List[PathPoint]
            List of points
        width : int
            Width of the frame
        height : int
            Height of the frame
        margin : int, optional
            Margin, by default 0

        Returns
        -------
        List[PathPoint]
            List of points inside the frame with the margin
        """

        return [
            point
            for point in path
            if point.center[0] > 0 - margin
            and point.center[1] > 0 - margin
            and point.center[0] < width + margin
            and point.center[1] < height + margin
        ]

    def draw(
        self,
        img: PIL.Image.Image,
        detection: norfair.Detection,
        coord_transformations,
        color: tuple = (255, 255, 255),
    ) -> PIL.Image.Image:
        """
        Draw the path

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        detection : norfair.Detection
            Detection
        coord_transformations : _type_
            Coordinate transformations
        color : tuple, optional
            Color of the path, by default (255, 255, 255)

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """

        self.add_new_point(detection=detection, color=color)

        if len(self.past_points) < 2:
            return img

        path = [
            PathPoint.from_abs_bbox(
                id=i,
                abs_point=point,
                coord_transformations=coord_transformations,
                alpha=i / (1.2 * self.path_length),
                color=self.color_by_index[i],
            )
            for i, point in enumerate(self.past_points)
        ]

        path_filtered = self.filter_points_outside_frame(
            path=path,
            width=img.size[0],
            height=img.size[0],
            margin=250,
        )

        img = self.draw_path_slow(img=img, path=path_filtered)
        img = self.draw_path_arrows(img=img, path=path)

        return img
