from typing import List

import numpy as np
from norfair import Detection

from inference.base_classifier import BaseClassifier


class InertiaClassifier:

    WINDOW = 1
    FIRST_N = 2

    def __init__(
        self,
        classifier: BaseClassifier,
        inertia: int = 20,
        mode: int = WINDOW,
    ):
        """

        Improves classification by using tracker IDs.
        It uses past classifications of the object to filter out noise.

        Parameters
        ----------
        classifier : BaseClassifier
            Classifier to use.
        inertia : int, optional
            Number of previous classifications to use, by default 20
        mode : int, optional
            Mode to use, by default WINDOW
        """
        self.inertia = inertia
        self.classifier = classifier
        self.classifications_per_id = {}
        self.mode = mode

    def add_first_classification_to_id(self, detection: Detection):
        """
        Add the first classification to the id.

        Parameters
        ----------
        detection : Detection
            Detection to add the classification to.
        """
        self.classifications_per_id[detection.data["id"]] = [
            detection.data["classification"]
        ]

    def add_new_clasification_to_id(self, detection: Detection):
        """
        Add a new classification to the existing id.

        Parameters
        ----------
        detection : Detection
            Detection to add the classification to.
        """
        self.classifications_per_id[detection.data["id"]].append(
            detection.data["classification"]
        )

    def should_classify(self, detection: Detection) -> bool:
        """
        Check if the detection should be classified.

        This improves performance for modes such as first_n. Because
        only at the first n detections of the id the classifier will be called.

        Parameters
        ----------
        detection : Detection
            Detection to check.

        Returns
        -------
        bool
            True if the detection should be classified.
        """

        if self.mode == InertiaClassifier.WINDOW:
            return True

        elif self.mode == InertiaClassifier.FIRST_N:

            if detection.data["id"] not in self.classifications_per_id:
                return True
            elif len(self.classifications_per_id[detection.data["id"]]) < self.inertia:
                return True
            else:
                return False

        raise ValueError("Invalid mode")

    def add_classification_to_window(self, detection: Detection):
        """
        Add a new classification using window mode.

        Parameters
        ----------
        detection : Detection
            Detection to add the classification to.
        """

        if detection.data["id"] not in self.classifications_per_id:
            self.add_first_classification_to_id(detection)

        elif len(self.classifications_per_id[detection.data["id"]]) < self.inertia:
            self.add_new_clasification_to_id(detection)

        elif len(self.classifications_per_id[detection.data["id"]]) == self.inertia:
            self.classifications_per_id[detection.data["id"]].pop(0)
            self.add_new_clasification_to_id(detection)

    def add_first_n_classification(self, detection: Detection):
        """
        Add a new classification using first n mode.

        Parameters
        ----------
        detection : Detection
            Detection to add the classification to.
        """

        if detection.data["id"] not in self.classifications_per_id:
            self.add_first_classification_to_id(detection)

        elif len(self.classifications_per_id[detection.data["id"]]) < self.inertia:
            self.add_new_clasification_to_id(detection)

    def add_new_clasifications(self, detections: List[Detection]):
        """
        Load internal dictionary with new classifications.

        Parameters
        ----------
        detections : List[Detection]
            Detections to add the classification to.
        """

        for detection in detections:

            if self.mode == InertiaClassifier.WINDOW:
                self.add_classification_to_window(detection)
            elif self.mode == InertiaClassifier.FIRST_N:
                self.add_first_n_classification(detection)

    def set_detections_classification(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """
        Set the detections classification to the mode of the previous classifications.

        Parameters
        ----------
        detections : List[Detection]
            Detections to set the classification to.

        Returns
        -------
        List[Detection]
            Detections with the classification set.
        """

        for detection in detections:
            previous_classifications = self.classifications_per_id[detection.data["id"]]
            detection.data["classification"] = max(
                set(previous_classifications), key=previous_classifications.count
            )

        return detections

    def predict_from_detections(
        self, detections: List[Detection], img: np.ndarray
    ) -> List[Detection]:
        """
        Predict the classification of the detections.

        Parameters
        ----------
        detections : List[Detection]
            Detections to predict the classification of.
        img : np.ndarray
            Image to predict the classification of.

        Returns
        -------
        List[Detection]
            Detections with the classification set.
        """

        # Filter detections for clasificiations
        detections_for_classification = [
            detection for detection in detections if self.should_classify(detection)
        ]

        detections_classified = self.classifier.predict_from_detections(
            detections=detections_for_classification,
            img=img,
        )

        # Add detections to internal dictionary
        self.add_new_clasifications(detections_classified)

        # Set detections classification
        detections = self.set_detections_classification(detections)

        return detections
