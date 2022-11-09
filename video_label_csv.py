# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:55:11 2022

@author: Mehmet


Bu scrip bir videonun üzernden her bir frame için labelların kordinantlarını alarak bir dosyaya kaydetmesi için oluşturulmuştur.
"""

import cv2 as cv
import numpy as np
import os


path=os.getcwd()


#videonun yüklenmesi
video_name="video2.mp4"
video=cv.VideoCapture(video_name)

#Modelin yüklenmesi
net = cv.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#video boyutları
width =int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height =int(video.get(cv.CAP_PROP_FRAME_HEIGHT))




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
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            label.append(str(classIDs[i]))
            label.append(str(x))
            label.append(str(y))
            label.append(str(w))
            label.append(str(h))
          
            
            
            
    else :
        label=[str(0),str(0),str(0),str(0),str(0)]
    return label     

    
    
    
    
    

l=0

conf=0.5

if video.isOpened()==False:
    print("Somethings went wrong: Video not found. Please check your video name or extension!")
    
    
    
save=open(path+'/'+video_name.split('.')[0]+'.csv', 'w')    
    
while video.isOpened():    #sonsuz döngü
     
     ret,frame=video.read()   #videodaki tüm frameler tek tek frame e gider frame bitince ret =False olur.
     
     if ret==True:
         img=frame
         
         print(f"processing on {l}. frame")
         labels=load_image(frame)
         
         k=0
         for label in labels:
             if (k+1)%5==0:
                 
                 if (k+1)==len(labels):
                     save.write(str(l)+" "+labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+"\n")
                     print("done!")
                 else:
                     save.write(str(l)+" "+labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+",")
                     print("çift kaynak")
             k=k+1
     
         l=l+1
         
        
     
        
             
            
     if cv.waitKey(1) & 0xFF == ord("q") or ret==False :
         break     
             
             
video.release() 
          
             
             
            