# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:43:07 2022

@author: Mehmet
pip install yolov7detect
"""

import time
import cv2
import numpy as np
import nvidia_smi
import yolov7

model=yolov7.load("yolov7.pt",size=512 ,device='cuda:0')

model.conf = 0.5  # NMS confidence threshold
model.iou = 0.5  # NMS IoU threshold

names=["person","bicycle","car","motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"]  


np.random.seed(28)
colors = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')



stream=cv2.VideoCapture("video.mp4")


nvidia_smi.nvmlInit()

deviceCount = nvidia_smi.nvmlDeviceGetCount()

while True:
   
    ret,frame = stream.read()
    if ret==False:
        print("Somethink went wrong about video. Check it !")
        break
    
    start_time =time.time()
    results = model(frame, size=512, augment=False)
    
    #results = detector.detect(frame)
    end_time=time.time()
    
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    
    gpu="Memory : ({:.2f}% free): {}(total) ".format( 100*info.free/info.total, info.total )
    gpu2="{} (free), {} (used)".format(info.free, info.used)
    fps=1/(end_time-start_time)
    
    cv2.putText(frame,gpu, (15,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    cv2.putText(frame,gpu2, (15,60), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    cv2.putText(frame,f"{fps:.3f} FPS " , (15,90), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    
    
    predictions = results.pred[0]
    boxes = predictions[:, :4].tolist() # x1, y1, x2, y2
    scores = predictions[:, 4].tolist()
    categories = predictions[:, 5].tolist()
    
    for i,box in enumerate(boxes):
        global x1,cls,conf
        
        x1,y1,x2,y2=box
        cls=int(categories[i])
        conf=round(scores[i],2)
    
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            #(0, 255, 0),
            (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
            2,
        )
        
        cv2.putText(frame, str(conf), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                    #(0,255,0),
                    (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
                    2)
        cv2.putText(frame, str(names[cls]), (int(x1-((x1-x2)*0.85)), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                    #(0,255,0),
                    (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
                    2)
    """
    
    for result in results:
        
        left, top, right, bottom, cls, conf = result
        if cls==0:
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
    """        
    cv2.imshow("Output Frame", frame)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
nvidia_smi.nvmlShutdown()    
stream.release()
cv2.destroyAllWindows()