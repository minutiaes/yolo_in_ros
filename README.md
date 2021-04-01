# YOLOinROS
Compact object detection capability for ROS2. As an object detector, YOLO is implemented via DNN module of OpenCV.

# Features
* In regard of [Darknet](https://pjreddie.com/darknet/) support of OpenCV, this package supports YOLOv2 to v4 and also corresponding Tiny YOLO versions.
* All necessary parameters are stored in a Launch File
* Prediction results are published in two different ways.  
  1. In a custom ROS [message](https://git.fh-aachen.de/kurhan/yolo_msg)
  2. In a human readable form as an `Image`

## ROS Topics
### Publishers
|   Topic Name  |  Message Type  |
|:-------------:|:--------------:|
| `camera/yolo` | `Image`        |
| `yolo/array`  | `YoloMsgArray` |

### Subscription
|          Topic Name         |    Message Type   |
|:---------------------------:|:-----------------:|
| `camera/image_undist_color` | `CompressedImage` |

## Getting Started
### Dependencies
* OpenCV  
To install:
  ```sh
  $ python3 -m pip install opencv-python
  ```
* CvBridge   
To install:  
  ```sh
  $ sudo apt install ros-eloquent-cv-bridge
  ```
* [yolo_msg](https://git.fh-aachen.de/kurhan/yolo_msg)  
To install:  
  Clone repository to `/your_workspace/src`  
  ```sh
  $ git clone https://git.fh-aachen.de/kurhan/yolo_msg.git
  ```
  Then, build your workspace  
  ```sh
  $ colcon build
  ```
* [camera_rgb](https://git.fh-aachen.de/kurhan/camera_rgb) or an alternative image publisher package  
To install:  
  Clone repository to `/your_workspace/src`  
  ```sh
  $ git clone https://git.fh-aachen.de/kurhan/camera_rgb.git
  ```
  Then, build your workspace  
  ```sh
  $ colcon build
  ```



### Installation
1. Clone repository to `/your_workspace/src`
    ```sh
    $ git clone https://git.fh-aachen.de/kurhan/yolo_in_ros.git
    ```
2. Build `/your_workspace` with `colcon`
   ```sh
   $ colcon build
   ```

## Usage
### Launch File 
It consist of parameters specific to YOLO
1. `config_file` contains structure of YOLO and hyperparameters.
    - `.cfg` file
    - It must be under `yolo_in_ros/config/`
2. `model_file` contains weights/parameters obtained during trained.
    - `.weight` file
    - It must be under `yolo_in_ros/config/`
3. `class_file` contains object names that YOLO CNN is trained for.
    - It must be a `.txt` file.
    - One class per line.
    - It must be under `yolo_in_ros/config/`
4. `conf_thresh_val` set the threshold to eliminate predictions with low score.
5. `nms_threshold` sets the threshold of Non-maximum Supression which eliminates predictions with low confidence score for same objects.
6. `width` and `height` set the size of input image of CNN and they must be multiple of 32.
7. `scale` is a factor which converts 8-bit($`[0, 255]`$) pixel values to floating numbers in a range of $`[0, 1.0]`$
8. `color` indicates that input image has 3 layers(RGB).

- Files of pretrained YOLO can be found [here](https://pjreddie.com/darknet/yolo/) and [there](https://github.com/AlexeyAB)
  - For this ROS package the files of Tiny YOLOv4, which is trained by [current developers](https://github.com/AlexeyAB) with COCO dataset, are supplied.





