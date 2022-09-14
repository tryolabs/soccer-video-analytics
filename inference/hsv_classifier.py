from typing import List

import cv2
import numpy as np

from inference.base_classifier import BaseClassifier


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
        """
        Check tuple format

        Parameters
        ----------
        a_tuple : tuple
            Tuple to check
        name : str
            Name of the tuple

        Returns
        -------
        tuple
            Tuple checked

        Raises
        ------
        ValueError
            If tuple is not a tuple
        ValueError
            If tuple is not a tuple of 3 elements
        ValueError
            If tuple elements are not integers
        """
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
        """
        Check tuple intervals

        Parameters
        ----------
        a_tuple : tuple
            Tuple to check
        name : str
            Name of the tuple

        Returns
        -------
        tuple
            Tuple checked

        Raises
        ------
        ValueError
            If first element is not between 0 and 179
        ValueError
            If second element is not between 0 and 255
        ValueError
            If third element is not between 0 and 255
        """

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
        """
        Check filter format

        Parameters
        ----------
        filter : dict
            Filter to check

        Returns
        -------
        dict
            Filter checked

        Raises
        ------
        ValueError
            If filter is not a dict
        ValueError
            If filter does not have a name
        ValueError
            If filter does not have a lower hsv
        ValueError
            If filter does not have an upper hsv
        ValueError
            If name is not a string
        ValueError
            If lower hsv doesnt have correct tuple format
        ValueError
            If upper hsv doesnt have correct tuple format
        """

        if type(filter) != dict:
            raise ValueError("Filter must be a dict")
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
        """
        Get HSV image

        Parameters
        ----------
        img : np.ndarray
            Image to convert

        Returns
        -------
        np.ndarray
            HSV image
        """
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    def apply_filter(self, img: np.ndarray, filter: dict) -> np.ndarray:
        """
        Apply filter to image

        Parameters
        ----------
        img : np.ndarray
            Image to apply filter to
        filter : dict
            Filter to apply

        Returns
        -------
        np.ndarray
            Filtered image
        """
        img_hsv = self.get_hsv_img(img)
        mask = cv2.inRange(img_hsv, filter["lower_hsv"], filter["upper_hsv"])
        return cv2.bitwise_and(img, img, mask=mask)

    def crop_img_for_jersey(self, img: np.ndarray) -> np.ndarray:
        """
        Crop image to get only the jersey part

        Parameters
        ----------
        img : np.ndarray
            Image to crop

        Returns
        -------
        np.ndarray
            Cropped image
        """
        height, width, _ = img.shape

        y_start = int(height * 0.15)
        y_end = int(height * 0.50)
        x_start = int(width * 0.15)
        x_end = int(width * 0.85)

        return img[y_start:y_end, x_start:x_end]

    def get_img_power(self, img: np.ndarray) -> float:
        """
        Get image power.

        Power is defined as the number of non black pixels of an image.

        Parameters
        ----------
        img : np.ndarray
            Image to get power of

        Returns
        -------
        float
            Image power
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(img)

    def set_power_in_filter(self, img: np.ndarray, filter: dict) -> dict:
        """
        Applies filter to image and saves the output power in the filter.

        Parameters
        ----------
        img : np.ndarray
            Image to apply filter to
        filter : dict
            Filter to apply

        Returns
        -------
        dict
            Filter with power
        """
        img_filtered = self.apply_filter(img, filter)
        img_filtered = self.crop_img_for_jersey(img_filtered)
        filter["power"] = self.get_img_power(img_filtered)
        return filter

    def predict_img(self, img: np.ndarray) -> str:
        """
        Gets the filter with most power on img and returns its name.

        Parameters
        ----------
        img : np.ndarray
            Image to predict

        Returns
        -------
        str
            Name of the filter with most power
        """
        for i, filter in enumerate(self.filters):
            self.filters[i] = self.set_power_in_filter(img, filter)

        max_power_filter = max(self.filters, key=lambda x: x["power"])
        return max_power_filter["name"]

    def predict(self, input_image: List[np.ndarray]) -> str:
        """
        Predicts the name of the team from the input image.

        Parameters
        ----------
        input_image : List[np.ndarray]
            Image to predict

        Returns
        -------
        str
            Predicted team name
        """

        if type(input_image) != list:
            input_image = [input_image]

        return [self.predict_img(img) for img in input_image]
