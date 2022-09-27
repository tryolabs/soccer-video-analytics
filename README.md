# track-soccer-possession

Demo on how to track soccer ball possession.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/CWnlGBVaRpQ/0.jpg)](https://www.youtube.com/watch?v=CWnlGBVaRpQ)


Detection for both players and ball was made using YOLOv5. The weights for the player detector are downloaded from Pytorch Hub using the default pre trained weights. The ball detector weights were trained using a custom dataset. For downloading the weights for the ball detector, click the following link.

>__Warning__: Our ball detector weights are trained on a very small dataset. The weights are not very accurate. For a bullet-proof soccer ball model, we encourage to search for other model.

The code is structure in two different folders, one for everything related to detection and classification called `inference` and another one for everything related to soccer called `soccer`.

Most of the files under `inference` folder are not specific for this problem and can be used for any other object detection problem. 

For tracking [Norfair](https://github.com/tryolabs/norfair) library was used. It was also used to draw the absolute path of the ball thanks to its `Motion Estimator` class.

The main loop can be found in the file `run.py`. It is a simple loop that reads the frames from the video and passes them to the detection and tracking functions. The results are then drawn on the frame and the frame is saved to a video file.
