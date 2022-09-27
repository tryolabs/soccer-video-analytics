# track-soccer-possession

Demo on how to track soccer ball possession.

Detection for both players and ball was made using YOLOv5. The weights for the player detector are downloaded from Pytorch Hub using the default pre trained weights. The ball detector weights were trained using a custom dataset. For downloading the weights for the ball detector, click the following link.

The script `run.py` is used to run the demo.

>__Note__: Be sure to set in the script your own video path and the path of the soccer ball weights.

The code is structure in two different folders, one for everything related to detection and classification called `inference` and another one for everything related to soccer called `soccer`.

Most of the files under `inference` folder are not specific for this problem and can be used for any other object detection problem. 

For tracking we used [Norfair](https://github.com/tryolabs/norfair) library. We also used the new version of Norfair to draw the absolute path of the ball.