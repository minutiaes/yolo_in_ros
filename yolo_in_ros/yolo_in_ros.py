#!/usr/bin/env python3
import os
import time
from threading import Thread
import queue
import cv2 as cv
import numpy as np

import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from rclpy.duration import Duration


class YoloInRos(Node):
    classes = []
    net = None
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def __init__(self, nms_threshold=0.4, width=416, height=416, scale=0.00392, color=1, mean=None):
        super().__init__("object_detection")

        # YOLO attributes
        self.mean = mean if mean else [0, 0, 0]
        self.backend = cv.dnn.DNN_BACKEND_DEFAULT
        self.target = cv.dnn.DNN_TARGET_CPU

        self.declare_parameter("config_file")
        self.config = self.get_parameter(name="config_file").get_parameter_value().string_value
        self.declare_parameter("model_file")
        self.model = self.get_parameter(name="model_file").get_parameter_value().string_value
        self.declare_parameter("class_file")
        self.classes = self.get_parameter(name="class_file").get_parameter_value().string_value
        self.declare_parameter("conf_thresh_val")
        self.conf_threshold = self.get_parameter(name="conf_thresh_val").get_parameter_value().double_value

        self.nms_threshold = nms_threshold
        self.width = width
        self.height = height
        self.scale = scale
        self.color = color
        self.mean = mean

        # ROS attributes
        self.publisher_output = self.create_publisher(Image, "camera/yolo", 1)
        self.publisher_marker = self.create_publisher(Marker, "camera/marker", 1)
        self.subscriber_camera = self.create_subscription(Image, "camera/image", self.get_frame, 1)
        self.frame = Image
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

    def generate_marker(self, class_id, confidence, left, top, width, height):
        if class_id == 0:
            msg = Marker()
            center = [left+width/2, top+height/2]

            msg.header.frame_id = "yolo_marker"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.ns = YoloInRos.classes[class_id]
            msg.id = 0
            msg.type = 1  # rectangle
            msg.action = 0  # modify
            msg.pose.position.x = (center[0]/640) - 0.5
            msg.pose.position.y = 0.0
            msg.pose.position.z = -(center[1]/480) + 0.5
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.scale.z = 0.1
            msg.color.a = confidence
            msg.color.r = 0.0
            msg.color.g = 1.0
            msg.color.b = 0.0
            msg.lifetime = Duration(nanoseconds=int(1000000/self.predictionsQueue_.getFPS())).to_msg()

            self.publisher_marker.publish(msg)
        else:
            pass

    def load_classes(self):
        if self.classes:
            with open(os.path.join(YoloInRos.__location__, self.classes), "rt") as f:
                YoloInRos.classes = f.read().rstrip("\n").split("\n")

    def load_network(self):
        YoloInRos.net = cv.dnn.readNetFromDarknet(os.path.join(YoloInRos.__location__, self.config),
                                             os.path.join(YoloInRos.__location__, self.model)
                                             )
        YoloInRos.net.setPreferableBackend(self.backend)
        YoloInRos.net.setPreferableTarget(self.target)
        global outNames
        outNames = YoloInRos.net.getUnconnectedOutLayersNames()

    def post_process(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        def draw_prediction(classId, conf, left, top, right, bottom):
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
            label = "{:.2f}".format(conf)

            assert (classId < len(YoloInRos.classes))
            label = "{}: {}".format(YoloInRos.classes[classId], label)

            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.rectangle(frame, (left, top - (labelSize[1] + baseLine)), (left + labelSize[0], top), (255, 255, 255),
                         cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        classIds = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, self.conf_threshold, self.nms_threshold)
            nms_indices = nms_indices[:, 0] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])

        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            draw_prediction(classIds[i], confidences[i], left, top, left + width, top + height)
            self.generate_marker(classIds[i], confidences[i], left, top, width, height)
            print("classId: {} \n confidence: {} \n left: {}\n top: {}\n width: {}\n height: {}"
                  .format(classIds[i], confidences[i], left, top, width, height))

    def start_detection(self):

        global process
        process = True

        global processedFramesQueue
        processedFramesQueue = queue.Queue()

        def processing_thread_body():
            # global processedFramesQueue, predictionsQueue, process
            futureOutputs = []
            while process:
                # Get a next frame
                frame = None
                try:
                    frame = self.framesQueue_.get_nowait()
                    self.framesQueue_.queue.clear()  # Skip the rest of frames
                except queue.Empty:
                    pass

                if not frame is None:
                    frameHeight = frame.shape[0]
                    frameWidth = frame.shape[1]
                    # Create a 4D blob from a frame.
                    inpWidth = self.width if self.width else frameWidth
                    inpHeight = self.height if self.height else frameHeight
                    blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=self.color, ddepth=cv.CV_8U)
                    processedFramesQueue.put(frame)

                    # Run a model
                    YoloInRos.net.setInput(blob, scalefactor=self.scale, mean=self.mean)
                    if YoloInRos.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                        frame = cv.resize(frame, (inpWidth, inpHeight))
                        YoloInRos.net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

                    outs = YoloInRos.net.forward(outNames)
                    self.predictionsQueue_.put(np.copy(outs))

                while futureOutputs and futureOutputs[0].wait_for(0):
                    out = futureOutputs[0].get()
                    self.predictionsQueue_.put(np.copy([out]))
                    del futureOutputs[0]

                try:
                    outs = self.predictionsQueue_.get_nowait()
                    frame = processedFramesQueue.get_nowait()
                    self.post_process(frame, outs)

                    msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    msg.header.frame_id = "camera"
                    self.publisher_output.publish(msg)
                    # self.get_logger().info("Network: {}, Camera: {}".format(self.predictionsQueue_.getFPS(), self.framesQueue_.getFPS()))
                except queue.Empty:
                    pass

        processing_thread = Thread(target=processing_thread_body)
        processing_thread.start()

        """process = False
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