# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:39:36 2022

@author: Mehmet

-Bu script yolo model çıkarmak için gerekli olan txt dosyasını çıkarmak için oluşturulmuştur.

"""

import os



img=[".jpg",".JPG",".png",".PNG",".bmp",".BMP"]


path=os.getcwd()

path_i=path+"\\"+"valid"


imglist=os.listdir(path_i)


file_train = open(path+'\\valid.txt', 'w')







for i in imglist:
    
    for j in img:
        if i.endswith(j):
           file_train.write("data/valid"+"/"+i+"\n")
         
            
