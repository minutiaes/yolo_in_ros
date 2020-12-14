# YOLOinROS
Compact object detection capability for ROS2. As an object detector, YOLO is implemented via DNN module of OpenCV  

## Topics:
- `/camera/image` - subscribed for input
- `/camera/yolo` - output of yolo predictions with bounding boxes are published 
- `/camera/marker` - markers for detected objects published

## Parameters:
They can be found in launch.py file
- `config_file` - name of configuration file of YOLO, `<your_config>.cfg` as string 
- `model_file` - name of weight file of YOLO, `<your_weights>.weights` as string
- `class_file` - name of class file for labeling YOLO predictions, `<your_classes.txt` as string
- `conf_thresh_val` - desired confidence threshold for prediction filtering, `<your_value>` as float

## TO DO:
• Display detected objects in Rviz2 via markers