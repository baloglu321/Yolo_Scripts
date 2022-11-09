
"""
Created on Thu Jun 23 15:49:56 2022

@author: Mehmet

-Bu script yolo formatında yazılmış txt dosyasından frameler içinde karşılık gelen clasları çizdiremk için oluşturulmuştur.
"""

import os
import time
import cv2
import numpy as np

#image_folder = "1"
text_folder = "txt"
not_txt=[]
#label_name=open(os.getcwd()+"\\"+"obj.txt","r")
#classes=label_name.read()
classes=["Obj","names","Here"]
folder=os.getcwd()


np.random.seed(5)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
#images = [name for name in os.listdir(os.path.join('.',image_folder))]
videos=[videos for videos in os.listdir(folder)]
texts = [name for name in os.listdir(os.path.join('.',text_folder))]
video_name=videos[0]
video=cv2.VideoCapture(video_name)

width =int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height =int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer=cv2.VideoWriter("video kaydı.mp4",cv2.VideoWriter_fourcc(*"mp4v"),30,(width,height))
i=0
if video.isOpened()==False:
    print("Hata")
    
while video.isOpened():    #sonsuz döngü
     
     ret,frame=video.read()   #videodaki tüm frameler tek tek frame e gider frame bitince ret =False olur.

     if ret==True:
         img=frame
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
           
             cv2.rectangle(img,(x1,y1),
                           (int(x1+w1),int(y1+h1)),
                           (int(colors[int(clasesId)][0]),
                            int(colors[int(clasesId)][1]),
                            int(colors[int(clasesId)][2])),3)
             
             cv2.putText(img, classes[int(clasesId)],
                         (x1,y1-5), cv2.FONT_HERSHEY_COMPLEX,
                         0.75, (int(colors[int(clasesId)][0]),
                               int(colors[int(clasesId)][1]),
                               int(colors[int(clasesId)][2])),2)
         cv2.imshow("image",img)
         writer.write(img)
         i+=1
         print(f"{i}frame")
         
         
         
         
     else:
         break
     
     
     if cv2.waitKey(1) &0xFF == ord("q"):
         break
     
     
video.release()    
cv2.destroyAllWindows()   


