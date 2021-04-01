"""Launch file of YOLOinROS"""
from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    """Executes YOLOinROS node with related parameters"""
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
                {"nms_threshold": 0.4},
                {"width": 416},
                {"height": 416},
                {"scale": 0.00392},
                {"color": 1}
            ]
        )
    ])
