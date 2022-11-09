# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:12:40 2022

@author: polat
"""

import os
import time
import cv2
import numpy as np


class OpenCVYoloDetector:
    def __init__(self, network_size=(512, 512), confidence=0.8, device_id=0):
        self.confidence = confidence
        self.network_size = network_size
        self.net = None
        self.output_layers = None
        self.net_initialized = False
        self.device_id = device_id

    def run(self):
        # Load YOLO

        cv2.cuda.setDevice(self.device_id)
        self.net = cv2.dnn.readNetFromDarknet(
            "yolov7-pf-14.cfg",
            "yolov7-pf-v14_last.weights",
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    def detect(self, frame):
        if not self.net_initialized:
            self.run()
            self.net_initialized = True
        height, width, channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, self.network_size, (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    # object detected
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    boxes.append((cx, cy, w, h))  # put all rectangle areas
                    class_ids.append(class_id)  # name of the object tha was detected
                    confidences.append(float(confidence))

        iou_thresh = 0.5
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, iou_thresh)
        engine_results = []
        for index in range(len(boxes)):
            if index in indices:
                cx, cy, w, h = boxes[index]
                top = int(cy - h / 2)
                right = int(cx + w / 2)
                bottom = int(cy + h / 2)
                left = int(cx - w / 2)
                engine_results.append(
                    (
                        left,
                        top,
                        right,
                        bottom,
                        int(class_ids[index]),
                        round(confidences[index], 2),
                    )
                )

        return engine_results
    
names=["person","forklift"]    
    
cap=cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"avc1")
np.random.seed(28)
colors = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')
resolution = (int(cap.get(3)), int(cap.get(4)))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("test-inf-v7.mp4", fourcc, fps, resolution)
detector = OpenCVYoloDetector(network_size=(512, 512), confidence=0.25)

width =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
i=0
while True:
        
    ret,frame=cap.read()
        
    if ret:
        
        start_time =time.time()
        
        results = detector.detect(frame)
        end_time=time.time()
        
        
        fps=1/(end_time-start_time)
        cv2.putText(frame,f"{fps:.3f} FPS {i}. frame" , (15,55), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
        for result in results:
            left, top, right, bottom, cls, conf = result

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                #(0, 255, 0),
                (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
                2,
            )
            cv2.putText(frame, str(conf), (left,top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                        #(0,255,0),
                        (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
                        2)
            #print(str(conf))
            
        out.write(frame)
        cv2.imshow("f", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    out.release()