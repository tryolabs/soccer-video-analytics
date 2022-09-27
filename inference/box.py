from typing import Tuple

import numpy as np


class Box:
    def __init__(self, top_left: Tuple, bottom_right: Tuple, img: np.ndarray):
        """
        Initialize Box

        Parameters
        ----------
        top_left : Tuple
            Top left corner of the box
        bottom_right : Tuple
            Bottom right corner of the box
        img : np.ndarray
            Image containing the box
        """
        self.top_left = top_left
        self.bottom_right = bottom_right

        # make tuples int
        self.top_left = (int(self.top_left[0]), int(self.top_left[1]))
        self.bottom_right = (int(self.bottom_right[0]), int(self.bottom_right[1]))

        self.img = self.cut(img.copy())

    def cut(self, img: np.ndarray) -> np.ndarray:
        """
        Cuts the box from the image

        Parameters
        ----------
        img : np.ndarray
            Image containing the box

        Returns
        -------
        np.ndarray
            Image containing only the box
        """
        return img[
            self.top_left[1] : self.bottom_right[1],
            self.top_left[0] : self.bottom_right[0],
        ]
