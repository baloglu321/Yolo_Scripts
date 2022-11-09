# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:15:11 2022

@author: Mehmet

Bu kod  kaynak projesindeki yanlış detectionları algılamak için tasarlanmıştır.

Bu aşamada kod:
    -1 adet kaynak detect ettiyse hata verecek ve görüntüsünü half_weld klasörüne kayıt edecek.
    -2 adet kaynak detectionu varsa verilen treshold değerine göre değerlendirip ikisi arasındaki 
    alan farkından yada boyut farkından yarım kaynak olabileceğini söyleyecek şekilde uyaracak ve bu görüntüleri her 15 
    karede bir wron_weld klasörüne söz konusu görüntüyü kayıt edecek.
    
    
    
"""
import math
import cv2
import os
#csv dosyasını okuma 
save=open("video.csv","r")
labels=save.read().strip().split('\n')
treshold=35

"""
okunan csv dosyasında yapılabilecek durum çıkarımları ve hesaplamalar bu kısımda yapıldı.
self.status 3 farklı çıkış veriyor;
-Geçerli frame için tahmin yok
-yarım kaynak
-Çift kaynak var

çift kaynak varsa 2 farklı kaynak tanımı ayrılıyor ve x,y,w,h tanımları objlere tanımlanıyor.

kullanılacaksa calc metodu çağırılıyor ve region,lenght ve height farkları ve bunların yüzdeleri hessaplanıp
region_dif,region_per,lenght_dif,lenght_per,height_dif,height_per değişkenlerine atanıyor.
"""

class Label:
    
    def __init__ (self,frame_number):
        
        label=labels[frame_number]

        if len(label.split(" "))<=6 :
            if  label.split(" ")[4]=='0' and  label.split(" ")[5]=='0':
                self.status="Geçerli frame için tahmin yok"
                
                
            else :
                self.status="Half Weld"
                
                        
        else:
            self.weld1=label.split(",")[0]
            self.weld2=label.split(",")[1]
            self.lenght1=int(self.weld1.split(" ")[4])
            self.lenght2=int(self.weld2.split(" ")[4])
            self.height1=int(self.weld1.split(" ")[5])
            self.height2=int(self.weld2.split(" ")[5])
            self.x1=int(self.weld1.split(" ")[2])
            self.x2=int(self.weld2.split(" ")[2])
            self.y1=int(self.weld1.split(" ")[3])
            self.y2=int(self.weld2.split(" ")[3])
            self.class_ıd1=int(self.weld1.split(" ")[1])
            self.class_ıd2=int(self.weld2.split(" ")[1])
            self.region1=(self.lenght1*self.height1)
            self.region2=(self.lenght2*self.height2)
            self.status="Çift kaynak var"


    def calc(self):
        if self.status=="Çift kaynak var":
            self.region_dif=math.sqrt((self.region1-self.region2)**2)
            self.lenght_dif=math.sqrt((self.lenght1-self.lenght2)**2)
            self.height_dif=math.sqrt((self.height2-self.height2)**2)
            if self.lenght1<=self.lenght2:
                self.lenght_per=(self.lenght_dif*100)/self.lenght2
            else:  
                self.lenght_per=(self.lenght_dif*100)/self.lenght1
                
            if self.height1<=self.height2:
                self.height_per=(self.height_dif*100)/self.height2
            else:  
                self.height_per=(self.height_dif*100)/self.height1
                
            if self.region1<=self.region2:
                self.region_per=(self.region_dif*100)/self.region2
            else:  
                self.region_per=(self.region_dif*100)/self.region1    
            
        else:
            print(self.status)
            


video_name="video.mp4"                                #video yükleme
video=cv2.VideoCapture(video_name)
width =int(video.get(cv2.CAP_PROP_FRAME_WIDTH))        #video boyutlarını alma
height =int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))



if video.isOpened()==False:
    print("Somethings went wrong: Video not found. Please check your video name or extension!") #video açılmadı ise hata ver
    
    
i=0                                                                            #Frame numaraları i ye atandı
k=0                                                                            #ardı ardına gelen görüntülerden fazla alınmaması için oluşturulan algoritma yazıldı. Bunlar için klm değişkenleri kullanıldı.   
l=0 
m=0
n=0

while video.isOpened():                                                        #sonsuz döngü
     
    ret,frame=video.read()                                                     #videodaki tüm frameler tek tek frame e gider frame bitince ret =False olur.
     
    if ret==True:
        image=Label(i) 
        print(f"processing on {i}. frame")                                                        #image yukardaki fonksiyon için tanımlandı.
        if image.status=="Geçerli frame için tahmin yok":
            
            if k%30==0:                                                         #detection alınmayan yerlerde resimlerin 30 da 1 ini al almamak için 0 yerine herhangi bir string ifade girilebilir.
                cv2.imwrite (f"no_detect/frame{str(i).zfill(7)}.png",frame)
                print(f"{i}. Frame no_detect kalsörüne kayıt edildi")
                k=k+1
                l=0
                m=0
                n=0
            else:
                k=k+1
                
        elif image.status=="Half Weld":                                         #yarım kaynak görüntüsü geldiğinde farmelerin 15 de birini al yükseltilebilir yada düşürülebilir.
            if l%5==0:
                cv2.imwrite (f"half_weld/frame{str(i).zfill(7)}.png",frame)
                print(f"{i}. Frame half_detect kalsörüne kayıt edildi")
                k=0
                l=l+1
                m=0
                n=0
            else:
                l=l+1
                
        else:
            image.calc()                                                       # görüntü yarım kaynak değilseve detection varsa alan ve uzunluk değerlerini hesapla
            
            if image.lenght_dif >= treshold:                         #bu görüntülerdeki x ekseninde değişim 20 pixelden fazla ise kaynak yarım olmalı. 
                                                                               #Alan farkı için image.region_dif>treshold yükseklik farkı için image.height_dif>treshold bunların yüzde farkları için _per kullanılmalı. hepsinin tresholdu yeniden incelenmeli.
                                                                               #proje özelinde en mantıklısı lenght_dif ve lenght_per olacaktır.
                if m%15==0:
                    cv2.imwrite (f"wrong_weld/frame{str(i).zfill(7)}.png",frame) #bu treshold dışında kalan görüntülerin 15 de 1 ini klasöre kaydet.
                    print(f"{i}. Frame wrong_weld kalsörüne kayıt edildi")
                    k=0
                    l=0
                    m=m+1
                    n=0
                else:
                    m=m+1
                
            else:                                                               #bu görüntülerdeki x ekseninde değişim 20 pixelden fazla ise kaynak yarım olmalı. 
                                                                              #Alan farkı için image.region_dif>treshold yükseklik farkı için image.height_dif>treshold bunların yüzde farkları için _per kullanılmalı. hepsinin tresholdu yeniden incelenmeli.
                                                                              #proje özelinde en mantıklısı lenght_dif ve lenght_per olacaktır.
               if n%30==0:
                   cv2.imwrite (f"okey/frame{str(i).zfill(7)}.png",frame) #bu treshold dışında kalan görüntülerin 15 de 1 ini klasöre kaydet.
                   print(f"{i}. Frame wrong_weld kalsörüne kayıt edildi")
                   k=0
                   l=0
                   m=0
                   n=n+1
               else:
                   n=n+1
        
    else:                                                                      #video bittiyse döngüyü bitir.
         break
     
     
    if cv2.waitKey(1) &0xFF == ord("q"):                                       #q harfine basılırsa döngüyü bitir.
         break     
     
    i=i+1                                                                      #frame bir artır.
    
             
video.release()                                                                #videoyu durdur.
wrong_weld=len(os.listdir("wrong_weld"))                                       #klasörlerdeki resim sayılarını al.
half_weld=len(os.listdir("half_weld"))
non_weld=len(os.listdir("no_detect"))
okey=len(os.listdir("okey"))
print(f"Frameler incelendi!\nSonuçlar;\nTespit olmayan detectionladan yakalanan görüntü sayısı :{non_weld}\nYarım kaynaktan yakalanan görüntü sayısı :{half_weld}\nHatalı kaynaktandan yakalanan görüntü sayısı :{wrong_weld}\nOkey görüntüden alınan görüntü sayısı :{okey} ")   
    
#klasörlerdeki resim sayılarını metin şeklinde çıktılarını yaz.   


        
        
        
        