from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from inference.box import Box


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:

        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------

        input_image: List[np.ndarray]
            List of input images

        Returns
        -------
        result: pd.DataFrame
            DataFrame containing the bounding boxes and the class of the objects


        The DataFrame must contain the following columns:
        - xmin: int
        - ymin: int
        - xmax: int
        - ymax: int
        - confidence: float
        - class: str
        """

        pass

    def check_result_format(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Checks if the result DataFrame has the correct format

        Parameters
        ----------
        result : pd.DataFrame
            DataFrame to check

        Returns
        -------
        pd.DataFrame
            DataFrame if it has correct format

        Raises
        ------
        TypeError
            If result type is not pd.DataFrame
        ValueError
            If result does not contain the correct columns
        """
        if type(result) != pd.DataFrame:
            raise TypeError("result must be a pandas DataFrame")

        if not {"xmin", "ymin", "xmax", "ymax"}.issubset(result.columns):
            raise ValueError("result must contain xmin, ymin, xmax, ymax columns")

        if not {"confidence", "class"}.issubset(result.columns):
            raise ValueError("result must contain confidence, class columns")

        return result

    def _draw_bounding_box(
        self,
        top_left: Tuple,
        bottom_right: Tuple,
        img: np.ndarray,
        color: Tuple = None,
        label: str = None,
    ) -> np.ndarray:
        """
        Draws a bounding box on the image

        Parameters
        ----------
        top_left : Tuple
            Top left corner of the bounding box (x, y)
        bottom_right : Tuple
            Bottom right corner of the bounding box (x, y)
        img : np.ndarray
            Image to draw the bounding box on
        color : Tuple, optional
            Color of the bounding box, by default None
        label : str, optional
            Label of the bounding box, by default None

        Returns
        -------
        np.ndarray
            Image with bounding box
        """

        if not color:
            color = (0, 255, 0)

        img = cv2.rectangle(img, top_left, bottom_right, color, 2)

        if label:
            img = cv2.putText(
                img,
                str(label),
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        return img

    @staticmethod
    def get_result_images(
        predictions: pd.DataFrame, img: np.ndarray
    ) -> List[np.ndarray]:
        """
        Returns a list of bounding box images

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame containing the bounding boxes and the class of the objects
        img : np.ndarray
            Image where the predictions were made

        Returns
        -------
        List[np.ndarray]
            List of bounding box images
        """
        images = []

        for index, row in predictions.iterrows():

            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            box = Box(top_left=(xmin, ymin), bottom_right=(xmax, ymax), img=img)
            images.append(box.img)

        return images

    @staticmethod
    def draw(self, predictions: pd.DataFrame, img: np.ndarray) -> np.ndarray:
        """
        Draws the bounding boxes on the image

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame containing the bounding boxes and the class of the objects
        img : np.ndarray
            Image where the predictions were made

        Returns
        -------
        np.ndarray
            Image with bounding boxes

        Raises
        ------
        TypeError
            If predictions type is not pd.DataFrame
        """

        if type(predictions) != pd.DataFrame:
            raise TypeError("predictions must be a pandas dataframe")

        for index, row in predictions.iterrows():

            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            # check if row has column team
            color = None
            if "color" in row:
                if not pd.isna(row["color"]):
                    color = row["color"]

            label = None

            if "label" in row:
                # check if pd is not nan
                if not pd.isna(row["label"]):
                    label = row["label"]

            # draw rect bounding box in img with cv2
            self._draw_bounding_box(
                top_left=(xmin, ymin),
                bottom_right=(xmax, ymax),
                img=img,
                color=color,
                label=label,
            )

        return img

    @staticmethod
    def generate_predictions_mask(
        predictions: pd.DataFrame, img: np.ndarray, margin: int = 0
    ) -> np.ndarray:
        """
        Generates a mask of the predictions bounding boxes

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame containing the bounding boxes and the class of the objects
        img : np.ndarray
            Image where the predictions were made
        margin : int, optional
            Margin to add to the bounding box, by default 0

        Returns
        -------
        np.ndarray
            Mask of the predictions bounding boxes

        Raises
        ------
        TypeError
            If predictions type is not pd.DataFrame
        """

        if type(predictions) != pd.DataFrame:
            raise TypeError("predictions must be a pandas dataframe")

        mask = np.ones(img.shape[:2], dtype=img.dtype)

        for index, row in predictions.iterrows():

            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            mask[ymin - margin : ymax + margin, xmin - margin : xmax + margin] = 0

        return mask
