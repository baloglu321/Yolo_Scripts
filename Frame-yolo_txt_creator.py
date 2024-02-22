# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:05:18 2022

@author: mehmet
"""


import os
import cv2 as cv
import numpy as np

path=os.getcwd()

image_folder = path+"/images"
txt_folder=path+"/txt"

img_list=os.listdir(image_folder)

net = cv.dnn.readNetFromDarknet('yolov4-custom.cfg', 'yolov4-custom_last.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def load_image(img):


    
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (512, 512), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(ln)

   
    outputs = np.vstack(outputs)

    label=post_process(img, outputs, 0.5)
   
    return label

def post_process(img, outputs, conf):
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

i=0
for image in img_list:
    img=cv.imread(image_folder+"/"+image)
    img_name=image.split(".")[0]
    resize_img=cv.resize(img,(512,512))
    print(f"processing on {i}. image")
    labels,na_label,confidences=load_image(resize_img)
    save=open(txt_folder+f"/{img_name}.txt", 'w')
    k=0
    for label in labels:
        if (k+1)%5==0:
            
                save.write(labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+"\n")
            
        k=k+1
    i=i+1
