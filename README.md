# crowd-counter

This is a short project made during a computer vision class I am taking in GWU.

To try out this program, first clone this repository. You are required to install `OpenCV` and `transformers` libraries in your Python environment.

```bash
$ pip install opencv-python
$ pip install transformers
```

Then, in the file directory, run:
```bash
$ python main.py <your-video-file-path>
```
where `<your-video-file-path>` is the file path for the video

## Problem Domain and Brief Project Description

The goal is to develop a system that detects and counts the number of people in a surveillance video or a time-lapse of a mall. This kind of system is important for crowd monitoring and resource management in public settings such as malls, airports and events.

The output of this system is the total number of individuals detected in the entire video, as well as noting down the number of individual in each frame.

Example Data Input/Output

> Input: A video file (e.g., mall_surveillance.mp4).

> Output: The total count of people detected in the video (e.g., "Total people detected in the video: 345").

