from typing import List, Union

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from inference.base_classifier import BaseClassifier


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batchnorm2(self.pool(x)))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(-1, 64 * 5 * 5)  # Flatten layer
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class NNClassifier(BaseClassifier):
    def __init__(
        self,
        model_path: str,
        classes: List[str] = None,
    ):
        """
        Initialize classifier

        Parameters
        ----------
        model_path : str, required
            Path to model .pt
        classes : List[str], optional
            List of class names, by default None
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = Net().to(self.device)  # On GPU
        self.model.load_state_dict(torch.load(model_path))

        self.classes = classes

    def convert_image_to_desired_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to desired tensor format for classification

        Parameters
        ----------
        image : np.ndarray
           Image to convert

        Returns
        -------
        torch.Tensor
            Image as tensor
        """

        pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = PIL.Image.fromarray(pil_image)

        pil_image = pil_image.resize((300, 300))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        pil_image = transform(pil_image).float()
        img_normalized = pil_image.unsqueeze_(0)
        img_normalized = img_normalized.to(self.device)

        return img_normalized

    def forward_image(self, image: torch.Tensor) -> Union[int, str]:
        """

        Predicts class of image and returns index or classname whether classes are defined

        Parameters
        ----------
        image : torch.Tensor
            image to predict

        Returns
        -------
        Union[int, str]
            index or classname
        """

        output = self.model(image)

        index = output.data.cpu().numpy().argmax()

        if self.classes:
            # check if index is in range
            if index < len(self.classes) and index >= 0:
                return self.classes[index]
            else:
                return "Unknown"

        return index

    def predict(self, input_image: List[np.ndarray]) -> List[Union[str, int]]:
        """

        Predict list of images and return list of predictions

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of images to predict

        Returns
        -------
        List[Union[str, int]]
            List of predictions
        """

        if not isinstance(input_image, list):
            input_image = [input_image]

        tensors = [
            self.convert_image_to_desired_tensor(input_image)
            for input_image in input_image
        ]

        with torch.no_grad():

            self.model.eval()

            result = [self.forward_image(tensor) for tensor in tensors]

            return result
