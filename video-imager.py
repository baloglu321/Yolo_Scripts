#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:43:25 2023

@author: mbaloglu
"""
import cv2 as cv
import numpy as np
import os


path=os.getcwd()



path_na="images/na-pred"
path_20="images/pred-20-40"
path_40="images/pred-40-60"
path_exist="images/exist"

os.makedirs(path_na,exist_ok=True)
os.makedirs(path_20,exist_ok=True)
os.makedirs(path_40,exist_ok=True)
os.makedirs(path_exist,exist_ok=True)


#Modelin yüklenmesi
net = cv.dnn.readNetFromDarknet('yolov4-custom.cfg', 'yolov4-custom_last.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]




def load_image(img,width,height):


    
    
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
            #x, y, w, h = output[:4] * np.array([W, H, W, H])
            x, y, w, h = output[:4]
            #p0 = int(x - w//2), int(y - h//2)
            #boxes.append([*p0, int(w), int(h)])
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

    
def save_na_pred(image,image_name):
    cv.imwrite(path_na+f"/frame_{image_name}.png", image)
    
def save_20_pred(image,image_name,labels):
    save=open(path_20+f"/frame_{image_name}.txt", 'w')
    cv.imwrite(path_20+f"/frame_{image_name}.png", image)   
    if len(labels)==0:
        save.write(" ")
    
    else:
        k=0
        for label in labels:
            if (k+1)%5==0:
                    
                    save.write(labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+"\n")
            k=k+1 
    


        
def save_40_pred(image,image_name,labels):
    save=open(path_40+f"/frame_{image_name}.txt", 'w')
    cv.imwrite(path_40+f"/frame_{image_name}.png", image)   
    if len(labels)==0:
        save.write(" ")
    
    else:
        k=0
        for label in labels:
            if (k+1)%5==0:
                    
                    save.write(labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+"\n")
            k=k+1 
  
          
def save_pred(image,image_name,labels):
    save=open(path_exist+f"/frame_{image_name}.txt", 'w')
    cv.imwrite(path_exist+f"/frame_{image_name}.png", image)   
    if len(labels)==0:
        save.write(" ")
    
    else:
        k=0
        for label in labels:
            if (k+1)%5==0:
                    
                    save.write(labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+"\n")
            k=k+1                           

def draw_img(image,labels,width,height):
    k=0
    for label in labels:
        
        if (k+1)%5==0:
                
                #save.write(labels[k-4]+" "+labels[k-3]+" "+labels[k-2]+" "+labels[k-1]+" "+labels[k]+"\n")
                
                x=float(labels[k-3])*width
                y=float(labels[k-2])*height
                w=float(labels[k-1])*width
                h=float(labels[k])*height
                                      
                    
              
                img=cv.rectangle(image,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0),2)
        k=k+1
        
    return img

video_list=os.listdir(path)
na_frame=0
pred_25_frame=0
pred_40_frame=0
exist_frame=0
for video_num,video in enumerate(video_list):
    if video.endswith(".mp4"):
        print(f"{video_num}. video processing please wait")
        cap=cv.VideoCapture(video)
        video_name=video.split(".")[0]
        width =int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height =int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        process=round((video_num/len(video_list))*100,3)
        if cap.isOpened()==False:
            print("Somethings went wrong: Video not found. Please check your video name or extension!")
            
        na=False
        pred_25=False
        pred_40=False
        pred_exist=False
        frame_number=0 
        na_label_count=0
        pred_25_label_count=0
        pred_40_label_count=0
        pred_exist_label=0
        while cap.isOpened():    #sonsuz döngü        
        
            ret,frame=cap.read()   #videodaki tüm frameler tek tek frame e gider frame bitince ret =False olur.
            width =int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height =int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


            
            if ret==True:
                
                if frame_number%2==0:
                    image=frame.copy()
                    labels,na_label,confidences=load_image(frame,width,height)
                    #mul=20
                    #count=0
                    if na_label :
                        if na==False or na_label_count==0:
                            save_na_pred(image, f"{video_name}_{str(frame_number).zfill(6)}")
                            na_frame+=1
                            print(f"image has been saved with {video_name}_{str(frame_number).zfill(6)} name succesfully")
                            print(f"Number of na_frame is {na_frame} ")
                            print(f"process complated % : {process}") 
                            na=True
                            na_label_count=1000
                            pred_25=False
                            pred_40=False
                            pred_exist=False
                            pred_25_label_count=0
                            pred_40_label_count=0
                            pred_exist_label=0
                        else:
                            na_label_count-=1

                    elif na_label==False:
                         conf=min(confidences)
                         na_label_count=0 
                         if 0.25<conf<0.45 :
                             if pred_25==False or  pred_25_label_count==0:
                                 print(f"conf = {conf}")
                                 save_20_pred(image,f"{video_name}_{str(frame_number).zfill(6)}",labels)
                                 pred_25_frame+=1
                                 print(f"image has been saved with {video_name}_{str(frame_number).zfill(6)} name succesfully")
                                 print(f"Number of pred 25 frame is {pred_25_frame} ")
                                 print(f"process complated % : {process}") 
                                 pred_25=True
                                 na=False
                                 pred_40=False
                                 pred_exist=False
                                 pred_25_label_count=5
                                 na_label_count=0
                                 pred_40_label_count=0
                                 pred_exist_label=0
                             else:
                                  pred_25_label_count-=1
                             
                         elif  0.45<conf<0.6 : 
                             if pred_40==False or  pred_40_label_count==0:
                                 print(f"conf = {conf}")
                                 save_40_pred(image,f"{video_name}_{str(frame_number).zfill(6)}",labels)
                                 pred_40_frame+=1
                                 print(f"image has been saved with {video_name}_{str(frame_number).zfill(6)} name succesfully")
                                 print(f"Number of pred 40 frame is {pred_40_frame} ")
                                 print(f"process complated % : {process}") 
                                 pred_25=False
                                 na=False
                                 pred_40=True
                                 pred_exist=False
                                 pred_40_label_count=10
                                 na_label_count=0
                                 pred_25_label_count=0
                                 pred_exist_label=0
                             else:
                                  pred_40_label_count-=1
                                  
                         elif 0.6<conf<1.0 :
                             if pred_exist==False or  pred_exist_label==0:
                                 print(f"conf = {conf}")
                                 exist_frame+=1
                                 print(f"Number of exist is {exist_frame} ")
                                 print(f"process complated % : {process}") 

                                 save_pred(image, f"{video_name}_{str(frame_number).zfill(6)}",labels)
                                 print(f"image has been saved with {video_name}_{str(frame_number).zfill(6)} name succesfully")
                                 pred_25=False
                                 na=False
                                 pred_40=False
                                 pred_exist=True
                                 pred_exist_label=15
                                 na_label_count=0
                                 pred_25_label_count=0
                                 pred_40_label_count=0
                             else:
                                  pred_exist_label-=1
                             #cv.destroyAllWindows()   
            
                          
            frame_number=frame_number+1
        
            if cv.waitKey(1) & 0xFF == ord("q") or ret==False :
                print("Process done!")

                break     

                 

                 
                            
                    
cap.release() 

cv.destroyAllWindows()             
                    
                    
                   
          
        
        
        
        
        
        
        
        
        