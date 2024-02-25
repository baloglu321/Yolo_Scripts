# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:54:53 2023

@author: polat
"""

import time
import cv2 as cv
import numpy as np
import nvidia_smi


#Modelin yüklenmesi

classes=["clases"]
color_green=(0,255,0)
color_red=(0,0,255)


def model_load(model_name,cfg):
    model_load_start=time.time()
    cv.cuda.setDevice(0)
    net = cv.dnn.readNetFromDarknet(cfg, model_name)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    model_load_stop=time.time()
    load_time=round((model_load_stop-model_load_start),2)
    print(f"Model yükleme süresi: {load_time} sn")
    
    layer_names = net.getLayerNames()
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return net,ln




def load_image(img,net,ln):


   
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (608, 608), swapRB=True, crop=False)

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
            #x, y, w, h = output[:4] * np.array([W, H, W, H])
            x, y, w, h = output[:4]
            #p0 = int(x - w//2), int(y - h//2)
            #boxes.append([*p0, int(w), int(h)])
            boxes.append([float(x),float(y), float(w), float(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            label.append(str(classIDs[i]))
            label.append(str(round(x,6)))
            label.append(str(round(y,6)))
            label.append(str(round(w,6)))
            label.append(str(round(h,6)))
          
            
            
            
    else :
        label=[]
    return label     

model_name=['yolov7_200000.weights','yolov7_100000.weights']
cfg=['yolov7_200000.cfg','yolov7_100000.cfg']
cap=cv.VideoCapture(0)
width =int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height =int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#şwidth =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


nvidia_smi.nvmlInit()

deviceCount = nvidia_smi.nvmlDeviceGetCount()

status=True

def sayac(start):
    status=False
    stop=time.time()
    if (stop-start) >= 15:
        status=True
    return status   

p=0

while True:
   
    #frame = stream.read()
    ret,frame=cap.read()
    image=cv.resize(frame,(1280,720))
    
    if ret:
          if status:
               start=time.time()
               print(p) 
               p=p+1
               if p%2==0:
                   
                   net,ln=model_load(model_name[0],cfg[0])
               else:
                   net,ln=model_load(model_name[1],cfg[1])
          status=sayac(start)
          start_time =time.time()
   
          labels= load_image(frame,net,ln)
                      
          #results = detector.detect(frame)
          end_time=time.time()
          k=0
          for label in labels:
              if len(labels)==0:
                  combined_img=frame.copy()
              elif (k+1)%5==0:
                      
                      #save.write(labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+"\n")
                      
                      cls=int(labels[k-4])
                      x=float(labels[k-3])*1280
                      y=float(labels[k-2])*720
                      w=float(labels[k-1])*1280
                      h=float(labels[k])*720
                                            
                          
                    
                      cv.rectangle(image,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,255,255),2)
                      combined_img=image.copy()
                      
              k=k+1
     
            #cv.imwrite(f"image/{str(frame_number).zfill(6)}.png",frame)
            #cv.imshow("image",image)

          #combined_img = model.draw_detections(frame)
          for i in range(deviceCount):
              handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
              info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
          
          gpu="Memory : ({:.2f}% free): {}(total) ".format( 100*info.free/info.total, info.total )
          gpu2="{} (free), {} (used)".format(info.free, info.used)
          fps=1/(end_time-start_time)
          
          
          cv.putText(combined_img,gpu, (15,30), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
          cv.putText(combined_img,gpu2, (15,60), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
          cv.putText(combined_img,f"{fps:.3f} FPS " , (15,90), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
          
          
          cv.imshow("Output Frame", combined_img)
       
    if cv.waitKey(1) & 0xFF == ord("q"):
            break
        
nvidia_smi.nvmlShutdown()    
cap.release()
#stream.stop()
cv.destroyAllWindows()