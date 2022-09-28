# Soccer Video Analytics

Demo on how to track soccer ball possession.

<a href="https://www.youtube.com/watch?v=CWnlGBVaRpQ">
<img src="https://raw.githubusercontent.com/tryolabs/soccer-video-analytics/main/images/thumbnail.png?token=GHSAT0AAAAAABU43Y7T7CL25XGPEKWDQA32YZUR2FA" width="60%" height="50%">
</a>

>__Note__:This repository was created in conjunction with this blog. If you want to know the details of the solution, I strongly encourage you to read the blog.

Detection for both players and ball was made using YOLOv5. The weights for the player detector are downloaded from Pytorch Hub using the default pre trained weights. The ball detector weights were trained using a custom dataset. For downloading the weights for the ball detector, click the following link.

>__Warning__: Our ball detector weights are trained on a very small dataset. The weights are not very accurate. For a bullet-proof soccer ball model, we encourage to search for other model.

The code is structured in two different folders, one for everything related to detection and classification called `inference` and another one for everything related to soccer called `soccer`.

Most of the files under `inference` folder are not specific for this problem and can be used for any other object detection problem. 

For tracking [Norfair](https://github.com/tryolabs/norfair) library was used. It was also used to draw the absolute path of the ball thanks to its `Motion Estimator` class.

The main loop can be found in the file `run.py`. It is a simple loop that reads the frames from the video and passes them to the detection and tracking functions. The results are then drawn on the frame and the frame is saved to a video file.
