from typing import List

import cv2
import numpy as np
import pandas as pd

from inference.base_classifier import BaseClassifier

# Chelsea
# Upper HSV:  (112, 255, 255)
# Lower HSV:  (0, 0, 0)

# City
# Upper HSV:  (91, 255, 255)
# Lower HSV:  (0, 0, 0)

# City GK
# Upper HSV:  (118, 255, 255)
# Lower HSV:  (35, 0, 0)

# Referee
# Upper HSV:  (179, 255, 51)
# Lower HSV:  (0, 0, 0)


class HSVClassifier(BaseClassifier):
    def __init__(self, filters: List[dict]):
        """
        Initialize HSV Classifier

        Parameters
        ----------
        filters : List[dict]
            List of filters to classify.

            Format:
            [
                {
                    "name": "Chelsea",
                    "lower_hsv": (112, 0, 0),
                    "upper_hsv": (179, 255, 255),
                },
                {
                    "name": "Man City",
                    "lower_hsv": (91, 0, 0),
                    "upper_hsv": (112, 255, 255),
                }
            ]
        """
        super().__init__()

        self.filters = [self.check_filter_format(filter) for filter in filters]

    def check_tuple_format(self, a_tuple: tuple, name: str) -> tuple:
        # Check upper hsv is a tuple
        if type(a_tuple) != tuple:
            raise ValueError(f"{name} must be a tuple")

        # Check lower hsv is a tuple of length 3
        if len(a_tuple) != 3:
            raise ValueError(f"{name} must be a tuple of length 3")

        # Check all lower hsv tuple values are ints
        for value in a_tuple:
            if type(value) != int:
                raise ValueError(f"{name} values must be ints")

    def check_tuple_intervals(self, a_tuple: tuple, name: str) -> tuple:

        # check hue is between 0 and 179
        if a_tuple[0] < 0 or a_tuple[0] > 179:
            raise ValueError(f"{name} hue must be between 0 and 179")

        # check saturation is between 0 and 255
        if a_tuple[1] < 0 or a_tuple[1] > 255:
            raise ValueError(f"{name} saturation must be between 0 and 255")

        # check value is between 0 and 255
        if a_tuple[2] < 0 or a_tuple[2] > 255:
            raise ValueError(f"{name} value must be between 0 and 255")

    def check_filter_format(self, filter: dict) -> dict:

        if "name" not in filter:
            raise ValueError("Filter must have a name")
        if "lower_hsv" not in filter:
            raise ValueError("Filter must have a lower hsv")
        if "upper_hsv" not in filter:
            raise ValueError("Filter must have an upper hsv")

        # Check name is a string
        if type(filter["name"]) != str:
            raise ValueError("Filter name must be a string")

        self.check_tuple_format(filter["lower_hsv"], "lower_hsv")
        self.check_tuple_format(filter["upper_hsv"], "upper_hsv")

        self.check_tuple_intervals(filter["lower_hsv"], "lower_hsv")
        self.check_tuple_intervals(filter["upper_hsv"], "upper_hsv")

        return filter

    def get_hsv_img(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    def apply_filter(self, img: np.ndarray, filter: dict) -> np.ndarray:
        img_hsv = self.get_hsv_img(img)
        mask = cv2.inRange(img_hsv, filter["lower_hsv"], filter["upper_hsv"])
        return cv2.bitwise_and(img, img, mask=mask)

    def get_img_power(self, img: np.ndarray) -> float:
        # convert hsv to grey
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(img)

    def set_power_in_filter(self, img: np.ndarray, filter: dict) -> dict:
        img_filtered = self.apply_filter(img, filter)
        filter["power"] = self.get_img_power(img_filtered)
        return filter

    def add_median_blur(self, img: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(img, 3)

    # predict img using filters dict and apply filter
    def predict_img(self, img: np.ndarray) -> str:
        for i, filter in enumerate(self.filters):
            self.filters[i] = self.set_power_in_filter(img, filter)

        max_power_filter = max(self.filters, key=lambda x: x["power"])
        return max_power_filter["name"]

    def predict(self, input_image: List[np.ndarray]) -> str:

        if type(input_image) != list:
            input_image = [input_image]

        return [self.predict_img(img) for img in input_image]
