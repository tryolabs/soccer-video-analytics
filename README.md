# Soccer Video Analytics


This repository contains the companion code of our blogpost on how to automatically track ball possession in soccer. For further information on the implementation, please check out the blogpost.

<a href="https://www.youtube.com/watch?v=CWnlGBVaRpQ" target="_blank">
<img src="https://user-images.githubusercontent.com/33181424/193869946-ad7e3973-a28e-4640-8494-bf899d5df3a7.png" width="60%" height="50%">
</a>

The scope of this repository is not to provide a production-ready solution, as we explained in the blogpost, many limitations need to be addressed before this is used in the real world. However, it can be used as a starting point for further research and development.


## How to install

To install the necessary dependencies we use [Poetry](https://python-poetry.org/docs). You can find the file `poetry.lock` in this project with all the dependencies.

The following steps describe the installation process.

- Clone this repository

```bash
git clone git@github.com:tryolabs/soccer-video-analytics.git
```

- Install [poetry](https://python-poetry.org) if you don't have it already.

```bash
pip install poetry
```

- Create virtualenv and install packages:

```bash
poetry config virtualenvs.in-project true
poetry install
```

## How to run

After installing the dependencies, you have to initialize the environment with these dependencies to run the project. This is achieved with the following command.

```
poetry shell
```

Once the environment is created, the possession and passes counter can be run.

To run one of these applications you need to use flags in the console.

These flags are defined in the following table:

| Argument | Description | Default value |
| ----------- | ----------- | ----------- |
| application | Set it to `possession` to run the possession counter or `passes` if you like to run the passes counter | None, but mandatory |
| path-to-the-model | Path to the soccer ball model weights (`pt` format) | `/models/ball.pt` |
| path-to-the-video | Path to the input video | `/videos/soccer_possession.mp4` |


The following command shows you how to run this project.

```
python run.py --<application> --model <path-to-the-model> --video <path-to-the-video>
```

>__Warning__: You have to run this command on the root of the project folder.

Here is an example on how to run the command:
    
```bash
python run.py --possession --model models/ball.pt --video videos/soccer_possession.mp4
```
An mp4 video will be generated after the execution. The name is the same as the input video with the suffix `_out` added.