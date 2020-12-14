from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_in_ros',
            node_executable='yolo_in_ros',
            node_name='camera1',
            output='screen',
            parameters=[
                {"config_file":"yolov4-tiny.cfg"},
                {"model_file":"yolov4-tiny.weights"},
                {"class_file":"object_detection_classes_yolov3.txt"},
                {"conf_thresh_val": 0.5},
            ]
        )
    ])
