#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:21:53 2023

@author: mbaloglu
Yolo v4 detection for coreset
"""

import cv2 as cv
import numpy as np
import os


#Modelin yüklenmesi
net = cv.dnn.readNetFromDarknet('yolov4-custom.cfg', 'yolov4-custom_last.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]




def load_image(img):

    height, width = img.shape[:2]
    
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (512, 512), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(ln)

   
    outputs = np.vstack(outputs)

    label=post_process(img, outputs, 0.25,width,height)
   
    return label

def post_process(img, outputs, conf,width,height):
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []
    label=[] 
    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
          
            x, y, w, h = output[:4]
     
            boxes.append([float(x),float(y), float(w), float(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
      

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            na_label=False
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            label.append(str(classIDs[i]))
            label.append(str(round(x,6)))
            label.append(str(round(y,6)))
            label.append(str(round(w,6)))
            label.append(str(round(h,6)))
          
            
            
            
    else :
        label=[]
        na_label=True
    return label,na_label,confidences   

def least_confidence_metric(softmax_scores):
    min_score = min(softmax_scores)
    least_confidence = 1 - min_score
    return least_confidence

def get_confidence_scores(results):

    least_confidences=[]
    for result in results:
        confidence=result[1]

        if len(confidence) == 0:
            least_confidence = 0
            least_confidences.append(least_confidence)
        else:
  
            softmax_scores = np.exp(confidence) / np.sum(np.exp(confidence))
            least_confidence = least_confidence_metric(softmax_scores)
            least_confidences.append(least_confidence)
    
    return least_confidences



def prune_objects_by_confidence(results, threshold):
    # Güven değerlerini al
    confidences = get_confidence_scores(results)

    # Güven değerlerine göre sırala
    sorted_results = [result for _, result in sorted(zip(confidences, results), key=lambda x: x[0], reverse=True)]

    # Kesme noktasını belirle
    num_to_keep = int(len(sorted_results) * threshold)

    # Belirlenen yüzdeyi al
    pruned_results = sorted_results[:num_to_keep]

    return pruned_results,sorted_results

def entropy_metric(softmax_scores):
    entropy = -np.sum(softmax_scores * np.log2(softmax_scores))
    return entropy

def get_entropy_scores(results):
    # Nesne çıktılarını al
    entropy_confidences=[]
    for result in results:
        confidence=result[1]

        if len(confidence) == 0:
            entropy_confidence = 0
            entropy_confidences.append(entropy_confidence)
        else:
            # Softmax fonksiyonunu uygula
            softmax_scores = np.exp(confidence) / np.sum(np.exp(confidence))
            entropy_confidence = entropy_metric(softmax_scores)
            entropy_confidences.append(entropy_confidence)
    
    return entropy_confidences        

def prune_objects_by_entropy(results, threshold):
    entropy_scores = get_entropy_scores(results)

    sorted_results = [result for _, result in sorted(zip(entropy_scores, results), reverse=True)]

    num_to_keep = int(len(sorted_results) * threshold)
    pruned_results = sorted_results[:num_to_keep]

    return pruned_results, sorted_results



image_directory = 'path to images'
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_paths = [os.path.join(image_directory, image_path) for image_path in os.listdir(image_directory) if image_path.lower().endswith(valid_extensions)]

num_images = len(image_paths)

engine_results = []
for image_path in image_paths:
    img = cv.imread(image_path)
    label, na_label, confidence = load_image(img)
    engine_results.append([image_path, confidence])

threshold = 0.5
pruned_results, sorted_results = prune_objects_by_confidence(engine_results, threshold)
pruned_results_entropy, sorted_results_entropy = prune_objects_by_entropy(engine_results, threshold)