#!/usr/bin/env python3
import os
import time
from threading import Thread
import queue
import cv2 as cv
import numpy as np

import rclpy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge
from yolo_msg.msg import YoloMsgFrame, YoloMsgArray


class YoloInRos(Node):
    """Creates a YOLO object detection object"""
    classes = []
    net = None
    __location__ = get_package_share_directory('yolo_in_ros')

    def __init__(self):
        super().__init__("object_detection")

        # YOLO attributes
        self.backend = cv.dnn.DNN_BACKEND_DEFAULT
        self.target = cv.dnn.DNN_TARGET_CPU

        self.declare_parameter("config_file")
        self.config = self.get_parameter(name="config_file").get_parameter_value().string_value
        self.declare_parameter("model_file")
        self.model = self.get_parameter(name="model_file").get_parameter_value().string_value
        self.declare_parameter("class_file")
        self.classes = self.get_parameter(name="class_file").get_parameter_value().string_value
        self.declare_parameter("conf_thresh_val")
        self.conf_threshold = self.get_parameter(
            name="conf_thresh_val").get_parameter_value().double_value
        self.declare_parameter("nms_threshold")
        self.nms_threshold = self.get_parameter(
            name="nms_threshold").get_parameter_value().double_value
        self.declare_parameter("width")
        self.width = self.get_parameter(name="width").get_parameter_value().integer_value
        self.declare_parameter("height")
        self.height = self.get_parameter(name="height").get_parameter_value().integer_value
        self.declare_parameter("scale")
        self.scale = self.get_parameter(name="scale").get_parameter_value().double_value
        self.declare_parameter("color")
        self.color = self.get_parameter(name="color").get_parameter_value().integer_value
        self.mean = [0, 0, 0]

        # ROS attributes
        self.publisher_output = self.create_publisher(Image, "camera/yolo", 1)
        self.publisher_array = self.create_publisher(YoloMsgArray, "yolo/array", 1)
        self.subscriber_camera = self.create_subscription(Image, "camera/image", self.get_frame, 1)
        self.bridge = CvBridge()

        self.framesQueue_ = self.QueueFPS()
        self.predictionsQueue_ = self.QueueFPS()

    class QueueFPS(queue.Queue):
        def __init__(self):
            queue.Queue.__init__(self, maxsize=1)
            self.startTime = 0
            self.counter = 0
        def put(self, v):
            queue.Queue.put(self, v)
            self.counter += 1
            if self.counter == 1:
                self.startTime = time.time()

        def getFPS(self):
            return self.counter / (time.time() - self.startTime)

    def get_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.framesQueue_.put(frame)

    def generate_yolo_msg(self, class_id, confidence, left, top, width, height):
        msg_frame = YoloMsgFrame()

        msg_frame.header.frame_id = "yolo_detection"
        msg_frame.header.stamp = self.get_clock().now().to_msg()
        msg_frame.object = str(YoloInRos.classes[class_id])
        msg_frame.prob = confidence
        msg_frame.bbox = [int(left), int(top), int(width), int(height)]

        return msg_frame


    def load_classes(self):
        if self.classes:
            with open(os.path.join(YoloInRos.__location__, 'config', self.classes), "rt") as f:
                YoloInRos.classes = f.read().rstrip("\n").split("\n")

    def load_network(self):
        YoloInRos.net = cv.dnn.readNetFromDarknet(os.path.join(YoloInRos.__location__, 'config', self.config),
                                             os.path.join(YoloInRos.__location__, 'config', self.model)
                                             )
        YoloInRos.net.setPreferableBackend(self.backend)
        YoloInRos.net.setPreferableTarget(self.target)
        global OUTNAMES
        OUTNAMES = YoloInRos.net.getUnconnectedOutLayersNames()

    def post_process(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        def draw_prediction(class_id, conf, left, top, right, bottom):
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
            label = "{:.2f}".format(conf)

            assert class_id < len(YoloInRos.classes)
            label = "{}: {}".format(YoloInRos.classes[class_id], label)

            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, label_size[1])
            cv.rectangle(
                frame,
                (left, top - (label_size[1] + base_line)),
                (left + label_size[0], top),
                (255, 255, 255),
                cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = []
        class_ids = np.array(class_ids)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(class_ids)
        for cl in unique_classes:
            class_indices = np.where(class_ids == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, self.conf_threshold, self.nms_threshold)
            nms_indices = nms_indices[:, 0] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])

        detection_array = YoloMsgArray()
        detection_array.data = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            draw_prediction(class_ids[i], confidences[i], left, top, left + width, top + height)
            detection_array.data.append(
                self.generate_yolo_msg(class_ids[i], confidences[i], left, top, width, height))
        self.publisher_array.publish(detection_array)


    def start_detection(self):

        process = True

        processed_frames_queue = queue.Queue()

        def processing_thread_body():

            future_outputs = []
            while process:
                # Get a next frame
                frame = None
                try:
                    frame = self.framesQueue_.get_nowait()
                    self.framesQueue_.queue.clear()  # Skip the rest of frames
                except queue.Empty:
                    pass

                if not frame is None:
                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]
                    # Create a 4D blob from a frame.
                    inp_width = self.width if self.width else frame_width
                    inp_height = self.height if self.height else frame_height
                    blob = cv.dnn.blobFromImage(frame, size=(inp_width, inp_height), swapRB=self.color, ddepth=cv.CV_8U)
                    processed_frames_queue.put(frame)

                    # Run a model
                    YoloInRos.net.setInput(blob, scalefactor=self.scale, mean=self.mean)
                    if YoloInRos.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                        frame = cv.resize(frame, (inp_width, inp_height))
                        YoloInRos.net.setInput(np.array([[inp_height, inp_width, 1.6]], dtype=np.float32), 'im_info')

                    outs = YoloInRos.net.forward(OUTNAMES)
                    self.predictionsQueue_.put(np.copy(outs))

                while future_outputs and future_outputs[0].wait_for(0):
                    out = future_outputs[0].get()
                    self.predictionsQueue_.put(np.copy([out]))
                    del future_outputs[0]

                try:
                    outs = self.predictionsQueue_.get_nowait()
                    frame = processed_frames_queue.get_nowait()
                    self.post_process(frame, outs)

                    msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    msg.header.frame_id = "camera"
                    self.publisher_output.publish(msg)
                    # self.get_logger().info("Network: {}, Camera: {}".format(self.predictionsQueue_.getFPS(), self.framesQueue_.getFPS()))
                except queue.Empty:
                    pass

        processing_thread = Thread(target=processing_thread_body)
        processing_thread.start()

        """processs = False
        processing_thread.join()"""


def main():
    rclpy.init()
    obj_det = YoloInRos()
    obj_det.load_classes()
    obj_det.load_network()
    obj_det.start_detection()
    try:
        rclpy.spin(obj_det)
    except KeyboardInterrupt:
        pass

    obj_det.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    