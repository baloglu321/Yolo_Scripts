# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:38:46 2022

@author: Mehmet

Bu script 1 isimli klasör içerisindeki tüm tiff dosyalarını aynı isimle jpeg olarak kaydetmek için oluşturulmuştur.
"""

import os
import cv2
import tifffile as tiff

image_folder = "1"

images = [name for name in os.listdir(os.path.join('.',image_folder))]

for index,image in enumerate(images):
    img=tiff.imread(image_folder+"\\"+image)
    cv2.imwrite(f"{str(index).zfill(10)}.jpeg",img,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
    print(f"{index}/{len(images)}")
    