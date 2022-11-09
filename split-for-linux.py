# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:27:09 2022

@author: mehmet

-Bu script çalıştırıldığı klasör içerisinden yolo formatında bulunan image ve txt dosyalarını ayrımak için oluşturulmuştur.

"""

import os
from math import *

path=os.getcwd()

img=[".jpg",".JPG",".png",".PNG",".bmp",".BMP"]
liste=os.listdir(path)
image_name=[]
txt_name=[]
for k in liste:
    for j in img:
        if k.endswith(j) :
            image_name.append(k)
    if k.endswith(".txt") :
        txt_name.append(k)

image_name=sorted(image_name)
txt_name=sorted(txt_name)


liste_b=len(image_name)
txt_b=len(txt_name)
image_number=int(input(f"{liste_b} adet resim var.{txt_b} adet txt dosyası var. Dosya başına istediğiniz resim sayısını giriniz :"))
file_number=floor(liste_b/image_number)
print(path)


for i in range (0,file_number+2):
    liste=os.listdir(path)
    image_name=[]
    txt_name=[]
    for k in liste:
        for j in img:
            if k.endswith(j) :
                image_name.append(k)
        if k.endswith(".txt") :
            txt_name.append(k)

    image_name=sorted(image_name)
    txt_name=sorted(txt_name)

    os.mkdir(path+'/'+str(i+1))
    os.mkdir(path+'/'+str(i+1)+'-txt')
    dizin=(path+'/'+str(i+1))
    dizin_txt=(path+'/'+str(i+1)+'-txt')


    if i==file_number+1:
        last_image_number=(len(image_name)-1)
        for l in range (0,last_image_number):
            if image_name[l].split(".")[0]==txt_name[l].split(".")[0]:
                os.rename(path+'/'+image_name[l], dizin+'/'+image_name[l])
                os.rename(path+'/'+txt_name[l], dizin_txt+'/'+txt_name[l])

    else:
        for m in range (0,image_number):
            if image_name[m].split(".")[0]==txt_name[m].split(".")[0]:
                os.rename(path+'/'+image_name[m], dizin+'/'+image_name[m])
                os.rename(path+'/'+txt_name[m], dizin_txt+'/'+txt_name[m])
                print("Source path renamed to destination path successfully.")