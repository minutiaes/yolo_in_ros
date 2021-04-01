import cv2 as cv
import numpy as np
import sys
import time
from threading import Thread
import queue
import os


class yolo:
    classes = []
    net = None
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def __init__(self, config: str, model: str, classes: str, conf_threshold=0.5, nms_threshold=0.4, width=416,
                 height=416, scale=0.00392, color=1, mean=None):
        self.mean = mean if mean else [0, 0, 0]
        self.backend = cv.dnn.DNN_BACKEND_DEFAULT
        self.target = cv.dnn.DNN_TARGET_CPU
        self.config = config
        self.model = model
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.width = width
        self.height = height
        self.scale = scale
        self.color = color
        self.mean = mean
        self.asyncN = 0

        self.freym_ = 1
        self.frame_raw = 0
        self.fremy_new = False
    
    class QueueFPS(queue.Queue):
        def __init__(self):
            queue.Queue.__init__(self)
            self.startTime = 0
            self.counter = 0

        def put(self, v):
            queue.Queue.put(self, v)
            self.counter += 1
            if self.counter == 1:
                self.startTime = time.time()

        def getFPS(self):
            return self.counter / (time.time() - self.startTime)

    def load_classes(self):
        if self.classes:
            with open(os.path.join(yolo.__location__, self.classes), "rt") as f:
                yolo.classes = f.read().rstrip("\n").split("\n")

    def load_network(self):
        yolo.net = cv.dnn.readNetFromDarknet(os.path.join(yolo.__location__, self.config),
                                             os.path.join(yolo.__location__, self.model)
                                             )
        yolo.net.setPreferableBackend(self.backend)
        yolo.net.setPreferableTarget(self.target)
        global outNames
        outNames = yolo.net.getUnconnectedOutLayersNames()

    def post_process(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        def draw_prediction(classId, conf, left, top, right, bottom):
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
            label = "{:.2f}".format(conf)

            assert (classId < len(yolo.classes))
            label = "{}: {}".format(yolo.classes[classId], label)

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

    def start_detection(self, input_source=0, input_frame=None):

        winName = 'Deep learning object detection in OpenCV'
        cv.namedWindow(winName, cv.WINDOW_NORMAL)
        cv.namedWindow("Input", cv.WINDOW_NORMAL)
        if input_source >= 0:
            cap = cv.VideoCapture(input_source)
        else:
            pass

        global framesQueue, process
        process = True
        framesQueue = self.QueueFPS()

        frame = False
        def framesThreadBody():
            # global framesQueue, process
            if input_source >= 0:
                while process:
                    hasFrame, frame = cap.read()
                    if not hasFrame:
                        
                        break
                    framesQueue.put(frame)
                    self.frame_raw = frame
            else:
                while process:
                    if not self.fremy_new:
                        pass
                    elif self.fremy_new:
                        frame = self.freym_
                        hasFrame = True
                        self.fremy_new = False
                        framesQueue.put(frame)

        global processedFramesQueue, predictionsQueue
        processedFramesQueue = queue.Queue()
        predictionsQueue = self.QueueFPS()

        def processingThreadBody():
            # global processedFramesQueue, predictionsQueue, process

            futureOutputs = []

            while process:
                # Get a next frame
                frame = None

                try:
                    frame = framesQueue.get_nowait()
                    if self.asyncN:
                        if len(futureOutputs) == self.asyncN:
                            frame = None  # Skip the frame
                    else:
                        framesQueue.queue.clear()  # Skip the rest of frames

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
                    yolo.net.setInput(blob, scalefactor=self.scale, mean=self.mean)
                    if yolo.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                        frame = cv.resize(frame, (inpWidth, inpHeight))
                        yolo.net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

                    if self.asyncN:
                        futureOutputs.append(yolo.net.forwardAsync())
                    else:
                        outs = yolo.net.forward(outNames)
                        predictionsQueue.put(np.copy(outs))

                while futureOutputs and futureOutputs[0].wait_for(0):
                    out = futureOutputs[0].get()
                    predictionsQueue.put(np.copy([out]))

                    del futureOutputs[0]

        framesThread = Thread(target=framesThreadBody)
        framesThread.start()
        time.sleep(1)
        processingThread = Thread(target=processingThreadBody)
        processingThread.start()

        #
        # Postprocessing and rendering loop
        #
        while cv.waitKey(1) < 0:
            try:
                # Request prediction first because they put after frames
                outs = predictionsQueue.get_nowait()
                frame = processedFramesQueue.get_nowait()

                self.post_process(frame, outs)

                # Put efficiency information.
                if predictionsQueue.counter > 1:
                    label = 'Camera: %.2f FPS' % (framesQueue.getFPS())
                    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                    label = 'Network: %.2f FPS' % (predictionsQueue.getFPS())
                    cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                    label = 'Skipped frames: %d' % (framesQueue.counter - predictionsQueue.counter)
                    cv.putText(frame, label, (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                cv.imshow(winName, frame)
                cv.imshow("Input",self.frame_raw)

            except queue.Empty:
                pass

        process = False
        framesThread.join()
        processingThread.join()


if __name__ == "__main__":
    obj_det = yolo(config="yolov4-tiny.cfg", model="yolov4-tiny.weights", classes="object_detection_classes_yolov3.txt",
                   conf_threshold=0.1)
    obj_det.load_classes()
    obj_det.load_network()
    obj_det.start_detection()

