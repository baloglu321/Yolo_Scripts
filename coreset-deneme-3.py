#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:01:09 2023

@author: mbaloglu
Çeşitli metodlarla coreset üretme çalışması.
Generating coresets with various methods.
"""
import cv2
import numpy as np
import os
import time
import shutil
from sklearn.decomposition import PCA


def euclidean_distance(feature1, feature2):
    feature1 = feature1.flatten()
    feature2 = feature2.flatten()
    return np.linalg.norm(feature1 - feature2)

def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    hist_flat = hist.flatten()
    return hist_flat

def print_with_logging(*args, **kwargs):
    # Gelen argümanları print fonksiyonuna yönlendir
    built_in_print(*args, **kwargs)
    
    # Çıktıları nohup.out dosyasına yönlendir
    with open('nohup2.out', 'a') as f:
        built_in_print(*args, file=f, **kwargs)

# Varsayılan print fonksiyonunu sakla
built_in_print = print


def extract_hog_features(image, cell_size=(8, 8), block_size=(4, 4), block_stride=(1, 1)):
    # Görüntüyü gri tonlamaya çevirme
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Görüntüyü yeniden boyutlandırma
    resized_image = cv2.resize(gray_image, (256, 256))
    
    # HOG öznitelik çıkarıcısını oluşturma
    win_size = (resized_image.shape[1], resized_image.shape[0])
    block_size = (block_size[0] * cell_size[0], block_size[1] * cell_size[1])
    block_stride = (block_stride[0] * cell_size[0], block_stride[1] * cell_size[1])
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 9)
    
    # HOG özniteliklerini hesaplama
    hog_features = hog.compute(resized_image)
    
    # Hesaplanan öznitelikleri düzleştirme
    hog_features = hog_features.flatten()
    
    return hog_features


def extract_sift_features(image, num_features=1000000, num_components=None):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    descriptors_flat = descriptors.flatten()

    # PCA uygulama
    if num_components is not None:
        pca = PCA(n_components=num_components)
        descriptors_flat = pca.fit_transform(descriptors_flat.reshape(1, -1))
        descriptors_flat = descriptors_flat.flatten()

    # Öznitelik vektörünü istenen boyuta getirme
    if num_features is not None:
        if len(descriptors_flat) > num_features:
            descriptors_flat = descriptors_flat[:num_features]
        else:
            descriptors_flat = np.pad(descriptors_flat, (0, num_features - len(descriptors_flat)))
            
          

    return descriptors_flat

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes} minutes {seconds} seconds"


def evaluate_pruning(image_paths, feature_extraction_method, threshold):
    pruned_image_indices = []
    not_pruned_image_indices = []
    total_elapsed_time = 0
    start_time = time.time()
    num_images = len(image_paths)
    images=[]

    print("Images uploaded")
    for image_path in image_paths:
        img=cv2.imread(image_path)
        images.append(img)
        
    image_upload_time=time.time()
    image_time=format_time(image_upload_time-start_time)
    print(f"Images uploaded :{image_time}") 
    built_in_print(f"Images uploaded :{image_time}")
    
    features = [feature_extraction_method(image) for image in images ]
    feature_extraction_time=time.time()
    feature_time=format_time(feature_extraction_time-image_upload_time)
    print(f"Feature extract :{feature_time}")
    built_in_print(f"Feature extract :{feature_time}")
    
    for i,feature in enumerate(features):
        loop_time = time.time() #perf_counter
        features_i = feature
        
        is_pruned = False
        for j,feature_j in enumerate(features[i+1:], start=i+1):
            features_j=feature_j
            distance = euclidean_distance(features_i, features_j)
            

            if distance < threshold:
                is_pruned = True
                break
                
        
        if is_pruned:
            pruned_image_indices.append(i)
        else:
            not_pruned_image_indices.append(i)

        # İlerleme gösterimi
        progress = (i + 1) / num_images

        

        progress = (i + 1) / num_images
        elapsed_time = time.time() - loop_time
        total_elapsed_time += elapsed_time
        total_elapsed_time_a = format_time(total_elapsed_time)
                
        average_time_per_image = total_elapsed_time / (i + 1)
        remaining_time = average_time_per_image * (num_images - i - 1)
        remaining_time = format_time(remaining_time)
        loop_fnish_time=time.time()
        loop_turn_time=format_time(loop_fnish_time-loop_time)
        print(f"Loop turned :{loop_turn_time}")
        built_in_print(f"Loop turned :{loop_turn_time}")
        
        print(f"Geçen Süre: {total_elapsed_time_a}")
        built_in_print(f"Geçen Süre: {total_elapsed_time_a}")
        
        print(f"Tahmini Kalan Süre: {remaining_time} ")
        built_in_print(f"Kalan Süre: {remaining_time} ")
        print(f"Progress: {progress*100:.2f}%")
        built_in_print(f"Progress: {progress*100:.2f}%")
        

    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = format_time(elapsed_time)
    print(f"Elapsed Time: {elapsed_time}")
    built_in_print(f"Elapsed Time: {elapsed_time}")
   

    return not_pruned_image_indices, pruned_image_indices
 
    
print = print_with_logging  


print("Process starting")
image_directory = 'train'
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_paths = [os.path.join(image_directory, image_path) for image_path in os.listdir(image_directory) if image_path.lower().endswith(valid_extensions)]


threshold_hist = 0.1
threshold_hog=8
threshold=43000



not_pruned_image_indices, pruned_image_indices = evaluate_pruning(image_paths, extract_color_histogram, threshold_hist)

print("Renk Histogramları ile Pruning Sonucu:")
print("Pruned Images:", len(pruned_image_indices))
print("Remaining Images:", len(not_pruned_image_indices))

pruned_image_paths = [image_paths[i] for i in pruned_image_indices]
not_pruned_image_paths = [image_paths[i] for i in not_pruned_image_indices]

# for i, image_path in enumerate(pruned_image_paths):
#     image = cv2.imread(image_path)
#     txt=image_path.split(".")[0]
#     os.makedirs("pruned_images_hist", exist_ok=True)  # Dizin mevcut değilse oluştur
#     cv2.imwrite(f"pruned_images_hist/image_{i}.png", image)
#     txt_path=f"pruned_images_hist/image_{i}.txt"
#     shutil.copy(f"{txt}.txt", txt_path)

# Remaining resimleri kaydetmek için:
for i, image_path in enumerate(not_pruned_image_paths):
    image = cv2.imread(image_path)
    txt=image_path.split(".")[0]
    os.makedirs("remaining_images_hist", exist_ok=True)  # Dizin mevcut değilse oluştur
    cv2.imwrite(f"remaining_images_hist/image_{i}.png", image)
    txt_path=f"remaining_images_hist/image_{i}.txt"
    shutil.copy(f"{txt}.txt", txt_path)





not_pruned_image_indices, pruned_image_indices = evaluate_pruning(image_paths, extract_hog_features, threshold_hog)

print("Hog metodu ile Pruning Sonucu:")
print("Pruned Images:", len(pruned_image_indices))
print("Remaining Images:", len(not_pruned_image_indices))

pruned_image_paths = [image_paths[i] for i in pruned_image_indices]
not_pruned_image_paths = [image_paths[i] for i in not_pruned_image_indices]

# for i, image_path in enumerate(pruned_image_paths):
#     image = cv2.imread(image_path)
#     txt=image_path.split(".")[0]
#     os.makedirs("pruned_images_hist", exist_ok=True)  # Dizin mevcut değilse oluştur
#     cv2.imwrite(f"pruned_images_hist/image_{i}.png", image)
#     txt_path=f"pruned_images_hist/image_{i}.txt"
#     shutil.copy(f"{txt}.txt", txt_path)

# Remaining resimleri kaydetmek için:
print("image saving")
for i, image_path in enumerate(not_pruned_image_paths):
    image = cv2.imread(image_path)
    txt=image_path.split(".")[0]
    os.makedirs("remaining_images_hog", exist_ok=True)  # Dizin mevcut değilse oluştur
    cv2.imwrite(f"remaining_images_hog/image_{i}.png", image)
    txt_path=f"remaining_images_hog/image_{i}.txt"
    shutil.copy(f"{txt}.txt", txt_path)




not_pruned_image_indices, pruned_image_indices = evaluate_pruning(image_paths, extract_sift_features, threshold)

print("Sıft metodu ile Pruning Sonucu:")
print("Pruned Images:", len(pruned_image_indices))
print("Remaining Images:", len(not_pruned_image_indices))

pruned_image_paths = [image_paths[i] for i in pruned_image_indices]
not_pruned_image_paths = [image_paths[i] for i in not_pruned_image_indices]

# for i, image_path in enumerate(pruned_image_paths):
#     image = cv2.imread(image_path)
#     txt=image_path.split(".")[0]
#     os.makedirs("pruned_images_hist", exist_ok=True)  # Dizin mevcut değilse oluştur
#     cv2.imwrite(f"pruned_images_hist/image_{i}.png", image)
#     txt_path=f"pruned_images_hist/image_{i}.txt"
#     shutil.copy(f"{txt}.txt", txt_path)

# Remaining resimleri kaydetmek için:
print("image saving")
for i, image_path in enumerate(not_pruned_image_paths):
    
    image = cv2.imread(image_path)
    txt=image_path.split(".")[0]
    os.makedirs("remaining_images_sift", exist_ok=True)  # Dizin mevcut değilse oluştur
    cv2.imwrite(f"remaining_images_sift/image_{i}.png", image)
    txt_path=f"remaining_images_sift/image_{i}.txt"
    shutil.copy(f"{txt}.txt", txt_path)

print("Processs end")     





