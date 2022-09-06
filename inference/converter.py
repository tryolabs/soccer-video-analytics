from typing import List

import norfair
import numpy as np
import pandas as pd


class Converter:
    @staticmethod
    def DataFrame_to_Detections(df: pd.DataFrame) -> List[norfair.Detection]:
        """
        Converts a DataFrame to a list of norfair.Detection

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the bounding boxes

        Returns
        -------
        List[norfair.Detection]
            List of norfair.Detection
        """

        detections = []

        for index, row in df.iterrows():
            # get the bounding box coordinates
            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            box = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymax],
                ]
            )

            # get the predicted class
            name = row["name"]
            confidence = row["confidence"]

            data = {
                "name": name,
                "p": confidence,
            }

            if "color" in row:
                data["color"] = row["color"]

            if "label" in row:
                data["label"] = row["label"]

            if "classification" in row:
                data["classification"] = row["classification"]

            detection = norfair.Detection(
                points=box,
                data=data,
            )

            detections.append(detection)

        return detections

    @staticmethod
    def Detections_to_DataFrame(detections: List[norfair.Detection]) -> pd.DataFrame:
        """
        Converts a list of norfair.Detection to a DataFrame

        Parameters
        ----------
        detections : List[norfair.Detection]
            List of norfair.Detection

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        df = pd.DataFrame()

        for detection in detections:

            xmin = detection.points[0][0]
            ymin = detection.points[0][1]
            xmax = detection.points[1][0]
            ymax = detection.points[1][1]

            name = detection.data["name"]
            confidence = detection.data["p"]

            data = {
                "xmin": [xmin],
                "ymin": [ymin],
                "xmax": [xmax],
                "ymax": [ymax],
                "name": [name],
                "confidence": [confidence],
            }

            # get color if its in data
            if "color" in detection.data:
                data["color"] = [detection.data["color"]]

            if "label" in detection.data:
                data["label"] = [detection.data["label"]]

            if "classification" in detection.data:
                data["classification"] = [detection.data["classification"]]

            df_new_row = pd.DataFrame.from_records(data)

            df = pd.concat([df, df_new_row])

        return df

    @staticmethod
    def TrackedObjects_to_Detections(
        tracked_objects: List[norfair.tracker.TrackedObject],
    ) -> List[norfair.Detection]:
        """
        Converts a list of norfair.tracker.TrackedObject to a list of norfair.Detection

        Parameters
        ----------
        tracked_objects : List[norfair.tracker.TrackedObject]
            List of norfair.tracker.TrackedObject

        Returns
        -------
        List[norfair.Detection]
            List of norfair.Detection
        """

        live_objects = [
            entity for entity in tracked_objects if entity.live_points.any()
        ]

        detections = []

        for tracked_object in live_objects:
            detection = tracked_object.last_detection
            detection.data["id"] = int(tracked_object.id)
            detections.append(detection)

        return detections
