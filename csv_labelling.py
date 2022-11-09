# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:13:48 2022

@author: Mehmet

Bu kod --- projesindeki yanlış detectionları algılamak için tasarlanmıştır.

Bu aşamada kod:
    -1 adet detection detect ettiyse hata verecek.
    -2 adet detection detectionu varsa verilen treshold değerine göre değerlendirip ikisi arasındaki 
    alan farkından yarım kaynak olabileceğini söyleyecek şekilde uyaracak ve bu görüntüleri her 20 
    karede bir dosyaya kayıt edecek.
    
"""

import math
#csv dosyasını okuma 
save=open("video.csv","r")
labels=save.read().strip().split('\n')

status=open("status.csv","w")


for label in labels:
    
    if len(label.split(" "))<=6:
        frame_number=label.split(" ")[0]
        class_ıd=label.split(" ")[1]
        x=label.split(" ")[2]
        y=label.split(" ")[3]
        w=label.split(" ")[4]
        h=label.split(" ")[5]
        if h=='0' and w=='0':    
            print(f"{frame_number} numaralı framede tespit yok ")
            status.write(str(frame_number)+",tespit yok \n")
        else :
            
            print(f"{frame_number} numaralı frame; kaynakta hata var")
            status.write(str(frame_number)+",kaynak eksik \n")
    else :
        weld1=label.split(",")[0]
        weld2=label.split(",")[1]
        frame_number1=weld1.split(" ")[0]
        class_ıd1=weld1.split(" ")[1]
        x1=int(weld1.split(" ")[2])
        y1=int(weld1.split(" ")[3])
        w1=int(weld1.split(" ")[4])
        h1=int(weld1.split(" ")[5])
        class_ıd2=weld2.split(" ")[1]
        x2=int(weld2.split(" ")[2])
        y2=int(weld2.split(" ")[3])
        w2=int(weld2.split(" ")[4])
        h2=int(weld2.split(" ")[5])
        
        lenght_dif=math.sqrt((w1-w2)**2)
        height_dif=math.sqrt((h1-h2)**2)
        region1=w1*h1
        region2=w2*h2
        region_dif=math.sqrt((region1-region2)**2)
        if w1<=w2:
            lenght_per=(lenght_dif*100)/w2
        else:  
            lenght_per=(lenght_dif*100)/w1
        
        if h1<=h2:
            height_per=(height_dif*100)/h2
        else:  
            height_per=(height_dif*100)/h1 
         
        if region1<=region2:
             region_per=(region_dif*100)/region2
        else:  
             region_per=(region_dif*100)/region1  
             
             
            
        print(f"{frame_number1} numaralı frame genişlik farkı % :{lenght_per} ")
        print(f"{frame_number1} numaralı frame yükseklik farkı % :{height_per} ")
        print(f"{frame_number1} numaralı frame alan farkı % :{region_per} ")
        status.write(str(frame_number1)+","+str(lenght_dif)+","+str(lenght_per)+"\n")    
