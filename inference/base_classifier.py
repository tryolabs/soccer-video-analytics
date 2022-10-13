import json
import os
from abc import ABC, abstractmethod
from typing import Counter, List, Tuple

import cv2
import norfair
import numpy as np
import pandas as pd

from inference.box import Box


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, input_image: List[np.ndarray]) -> List[str]:
        """
        Predicts the class of the objects in the image

        Parameters
        ----------

        input_image: List[np.ndarray]
            List of input images

        Returns
        -------
        result: List[str]
            List of class names
        """
        pass

    def predict_from_df(self, df: pd.DataFrame, img: np.ndarray) -> pd.DataFrame:
        """
        Predicts the class of the objects in the image and adds a column
        in the dataframe for classification

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the bounding boxes
        img : np.ndarray
            Image

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes and the class of the objects

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        box_images = []

        for index, row in df.iterrows():

            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            box = Box((xmin, ymin), (xmax, ymax), img)

            box_images.append(box.img)

        class_name = self.predict(box_images)

        df["classification"] = class_name

        return df

    def predict_from_detections(
        self, detections: List[norfair.Detection], img: np.ndarray
    ) -> List[norfair.Detection]:
        """
        Predicts the class of the objects in the image and adds the class in
        detection.data["classification"]

        Parameters
        ----------
        detections : List[norfair.Detection]
            List of detections
        img : np.ndarray
            Image

        Returns
        -------
        List[norfair.Detection]
            List of detections with the class of the objects
        """
        if not all(
            isinstance(detection, norfair.Detection) for detection in detections
        ):
            raise TypeError("detections must be a list of norfair.Detection")

        box_images = []

        for detection in detections:
            box = Box(detection.points[0], detection.points[1], img)
            box_images.append(box.img)

        class_name = self.predict(box_images)

        for detection, name in zip(detections, class_name):
            detection.data["classification"] = name

        return detections

    def accuarcy_on_folder(
        self, folder_path: str, label: str
    ) -> Tuple[float, List[np.ndarray]]:
        """
        Calculates the accuracy of the classifier on a folder

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the images of the same label
        label : str
            Label of the images in the folder

        Returns
        -------
        float
            Accuracy of the classifier
        List[np.ndarray]
            List of the images that were misclassified
        """
        images = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)

        predictions = self.predict(images)

        missclassified = [images[i] for i, x in enumerate(predictions) if x != label]

        counter = Counter()
        for prediction in predictions:
            counter[prediction] += 1

        print(json.dumps(counter, indent=4))

        return counter[label] / len(predictions), missclassified
