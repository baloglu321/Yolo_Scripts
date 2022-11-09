# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:52:11 2022

@author: mehmet

Bu kod yolov4 darknet üzerinde farklı klasörler üzerinden valid okuyarak map hesaplamalarını çıkarması için oluşturulmuştur.

darknet ana klasörü içerisinde çalıştılması gerekmektedir. 

data klsöründe "valid", "weights", "valid-text", "maps" klasörleri oluşturunuz.

data/valid içerisinde map hesaplaması yapılacak olan klsör içerindede valid datalarının yolo formatında bulunması gerekir.

data/weights içersinde yolov4 weights ve cfg dosyalarının bulunması gerekmektedir.

bu şartlar bulunduğu taktirde script weights klasöründe bulunan her weights ve cfg dosyası için valid klasöründeki verilerden map değeri hesaplayıp
klasör ismi ile main konum/result içine "result-(valid klasör ismi)-(weights dosyası).txt" şeklinde map değerlerini içeren dosyaları oluşturacaktır. 

"""

import os




path=os.getcwd()
data_path=path+'/data'
valid_path=data_path+'/valid'
valid_text_path=data_path+'/valid-text'
weights_path=data_path+'/weights'
map_path=data_path+'/maps'


file_name=os.listdir(valid_path)

valid_text_list=os.listdir(valid_text_path)
sorted(valid_text_list)

model_list=os.listdir(weights_path)

weights=[]
cfgs=[]
for model in model_list:
    if model.endswith('.weights'):
        weights.append(model)
    elif model.endswith('.cfg'):
        cfgs.append(model)
        

maps=[]
def Valid_txt(valid_list,file):
    
    img=[".jpg",".JPG",".png",".PNG",".bmp",".BMP","jpeg","JPEG"]
    file_valid=open(data_path+f'/valid-text/valid_{file}_.txt','w')  
    
    
    
    for k in valid_list:
        
        for l in img:
            if k.endswith(l):
               file_valid.write("data/valid"+'/'+file_name[p]+"/"+k+"\n")
    
        
for p in range(len(file_name)):
    
    file=file_name[p]
    
    
    
    
    image_path=valid_path+'/'+file_name[p]
        
    
    valid_list=os.listdir(image_path)
    
   
    Valid_txt(valid_list, file)
    
    
   
               
    obj_data=open(map_path+f'/map_{file}_.data','w')
    obj_data.write("classes = 2"+"\n"+f'valid = data/valid-text/valid_{file}_.txt'+"\n"+"names = data/obj.names"+"\n"+"backup = backup/")           
    maps.append(map_path+f'/map_{file}_.data')

obj_data=open(map_path+f'/map_{file}_.data','w')
obj_data.write("classes = 2"+"\n"+f'valid = data/valid-text/valid_{file}_.txt'+"\n"+"names = data/obj.names"+"\n"+"backup = backup/")           
maps.append(map_path+f'/map_{file}_.data')


def Map(cfg,weight_folder,file,weigt_name,valid_text,current_map)  :
    
    os.system(f'darknet.exe detector map {current_map} {cfg} {weight_folder} -dont_show -ext_output < data/valid-text/{valid_text} > results/result-{file}-{weight_name}.txt')                
    print("\ndone!")

n=0


for file in file_name:
    for a in maps:
        if a.split("_")[1]==file:
            current_map=a
            print(f"{current_map}")
        
    i=0
  
    for weight in weights:
           
        weight_name=weight.split('.')[0]
        cfg='data/weights/'+cfgs[i]
        weight_folder='data/weights/'+weight
        valid_text=valid_text_list[n]
        a=Map(cfg,weight_folder,file,weight_name,valid_text,current_map)
         
    n=+1                 
                   