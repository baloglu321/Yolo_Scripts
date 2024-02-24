#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:13:24 2023

@author: mbaloglu
Yolo formatındaki bir dataseti VOC formatına çevirmek için 
To convert a dataset in Yolo format to VOC format
"""
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom
from PIL import Image
import cv2

# Girdi klasörlerini ve çıkış klasörünü belirtin
input_image_folder = "images/"
input_label_folder = "labels/"
output_folder = "VOC_dataset/"
#clases = ["person", "forklift"]
clases = ['aeroplane','bicycle','bird','boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor','thing']
    

def convert_to_voc_format(image_name, label_path):
    global clases
    # Resim boyutlarını alın
    image_path = os.path.join(input_image_folder, image_name)
    image_width, image_height = get_image_size(image_path)

    # VOC formatında XML dosyasını oluşturun
    annotation = Element("annotation")
    SubElement(annotation, "folder").text = output_folder
    SubElement(annotation, "filename").text = image_name

    size_element = SubElement(annotation, "size")
    SubElement(size_element, "width").text = str(image_width)
    SubElement(size_element, "height").text = str(image_height)

    with open(label_path, 'r') as label_file:
        for line in label_file:
            print(line)
            class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.strip().split())

            xmin = int((center_x - bbox_width / 2) * image_width)
            ymin = int((center_y - bbox_height / 2) * image_height)
            xmax = int((center_x + bbox_width / 2) * image_width)
            ymax = int((center_y + bbox_height / 2) * image_height)

            object_element = SubElement(annotation, "object")
            SubElement(object_element, "name").text = clases[int(class_id)]
            SubElement(object_element, "pose").text = "Unspecified"
            SubElement(object_element, "truncated").text = "0"
            SubElement(object_element, "difficult").text = "0"

            bndbox_element = SubElement(object_element, "bndbox")
            SubElement(bndbox_element, "xmin").text = str(xmin)
            SubElement(bndbox_element, "ymin").text = str(ymin)
            SubElement(bndbox_element, "xmax").text = str(xmax)
            SubElement(bndbox_element, "ymax").text = str(ymax)

    # XML dosyasını kaydedin
    xml_string = minidom.parseString(tostring(annotation)).toprettyxml(indent="  ")
    output_xml_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".xml")
    with open(output_xml_path, "w") as xml_file:
        xml_file.write(xml_string)

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def main():
    # Çıkış klasörünü oluşturun
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_image_folder):
        if image_name.split(".")[-1] !="jpg":
            img=cv2.imread(f"images/{image_name}")
            cv2.imwrite("images/"+image_name.split(".")[0]+".jpg" , img)
            os.rename(f"images/{image_name}",f"temp_file/{image_name}" )
    # Resimleri dolaşarak VOC formatına çevirin
    
    image_list = os.listdir(input_image_folder)
    

    for image_name in image_list:

        label_name =image_name.split(".")[0]+".txt"
        label_path = os.path.join(input_label_folder, label_name)
        if os.path.isfile(label_path):

            convert_to_voc_format(image_name, label_path)

    print("VOC formatına dönüşüm tamamlandı.")

if __name__ == "__main__":
    main()
