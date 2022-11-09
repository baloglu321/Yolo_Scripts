# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:34:24 2022

@author: Mehmet

-Bu script frameler için oluşturulan yolo formatındaki text dosyasından labelları okuyaraklabelları çizip kaydetmesi için oluşturulmuştur.

"""

import os
import cv2
import numpy as np

image_folder = "1"
text_folder = f"{image_folder}-txt"
not_txt=[]
label_name=open(os.getcwd()+"\\"+"obj.txt","r")
#classes=label_name.read()
classes=["takoz","baret","forklift","person","yelek"]


np.random.seed(5)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
images = [name for name in os.listdir(os.path.join('.',image_folder))]
texts = [name for name in os.listdir(os.path.join('.',text_folder))]

for i in range(len(images)):
    img=cv2.imread("1"+"\\"+images[i])
    file=open(text_folder+"\\"+texts[i], 'r')
    #box=file.read()
    boxes=file.read().strip().split('\n')
    for box in boxes:
        clasesId,x,y,w,h=box.split(" ")
        H,W,_=img.shape
        x1=float(x)*W
        y1=float(y)*H
        w1=float(w)*W
        h1=float(h)*H
        x1=int(x1-w1//2)
        y1=int(y1-h1//2)
      
        cv2.rectangle(img,(x1,y1),(int(x1+w1),int(y1+h1)),(int(colors[int(clasesId)][0]),int(colors[int(clasesId)][1]),int(colors[int(clasesId)][2])),2)
        cv2.putText(img, classes[int(clasesId)], (x1,y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (int(colors[int(clasesId)][0]),int(colors[int(clasesId)][1]),int(colors[int(clasesId)][2])),1)
    cv2.imshow("image",img)
    print(images[i])
    cv2.waitKey(0)
    img_name="detect_"+images[i]
    cv2.imwrite(img_name,img)
    
    if cv2.waitKey(1) & 0xFF == ord("q") or i==(len(images)-1):
        break 
    
cv2.destroyAllWindows()
     