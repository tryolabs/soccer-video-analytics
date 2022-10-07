# Soccer Video Analytics

Demo on how to track soccer ball possession.

<a href="https://www.youtube.com/watch?v=CWnlGBVaRpQ" target="_blank">
<img src="https://user-images.githubusercontent.com/33181424/193869946-ad7e3973-a28e-4640-8494-bf899d5df3a7.png" width="60%" height="50%">
</a>


>__Note__: This repository was created in conjunction with this blog. If you want to know the details of the solution, I strongly encourage you to read the blog.

Detection for both players and ball was made using YOLOv5. The weights for the player detector are downloaded from Pytorch Hub using the default pre trained weights. The ball detector weights were trained using a custom dataset. To download the weights of the ball detector, click this link.

>__Warning__: Our ball detector weights are trained on a very small dataset. The weights are not very accurate. For a bullet-proof soccer ball model, we encourage to search for other model.

The code is structured in two different folders, one for everything related to detection and classification called `inference` and another one for everything related to soccer called `soccer`.

Most of the files under `inference` folder are not specific for this problem and can be used for any other object detection problem. 

We used [Norfair](https://github.com/tryolabs/norfair) as our tracking library which also enabled us to draw the absolute path of the ball thanks to its `Motion Estimator` class.

The main loop can be found in the file `run.py`. It is a simple loop that reads the frames from the video and passes them to the detection and tracking functions. The results are then drawn on the frame and the frame is saved to a video file.

## How to install

To install the necessary dependencies we use [Poetry](https://python-poetry.org/docs). You can find a `poetry.lock` in this project with all dependencies.

The following steps describe the installation process.

- Clone this repository

- Install [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org) if you don't have them already.

- Create virtualenv and install packages:

```
# (Optional, recommended)
poetry config virtualenvs.in-project true

poetry install
```

## How to run

After installing the dependencies, you have to initialize the environment with these dependencies to run the project. This is achieved with the following command.

```
poetry shell
```

Once the environment was created, the possession and passes counter could be run.

To run one of these applications you need to use flags in the console.

These flags are defined in the following chart.

| Argument | Description | Default value |
| ----------- | ----------- | ----------- |
| application | Set it to `possession` to run the possession counter or `passes` if you like to run the passes counter | None, but mandatory |
| path-to-the-model | Path to the model weights (`pt` format) | `/models/ball.pt` |
| path-to-the-video | Path to the input video | `/videos/soccer_possession.mp4` |


The following command shows you how to run this project.

```
python run.py --<application> --model <path-to-the-model> --video <path-to-the-video>
```

>__Warning__: You have to run this command on the root of the project folder.

After the execution will be generated a file with the same name as the input video concatenated with `_out` indicating that this is the output file.