# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:56:26 2022

@author: mehmet
"""

import numpy as np
import onnxruntime as ort
import cv2

from tool.utils import *
from tool.darknet2onnx import *
import time
from vidgear.gears import CamGear
import nvidia_smi


w="yolov4_8_3_512_512_static.onnx"

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

names="coco.names"
cfg_file="yolov4-coco.cfg"
weight_file="yolov4-coco.weights"
batch_size=8


onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)


session = ort.InferenceSession(onnx_path_demo, providers=providers)


#session = ort.InferenceSession(onnx_path_demo, providers=providers)




colors = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')



options = {"CAP_PROP_FRAME_WIDTH ":512, "CAP_PROP_FRAME_HEIGHT":512, "CAP_PROP_FPS ":60}




def detect(session, image_src, namesfile):
    
    
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    #◘print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)

    class_names = load_class_names(namesfile)
    img=plot_boxes_cv2(image_src, boxes[0], class_names=class_names)
    
    return img


np.random.seed(28)
colors = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')




stream = CamGear(source="https://www.youtube.com/watch?v=jg24Zo-a2f0",stream_mode=True, 
                time_delay=1, logging=True, **options).start()


nvidia_smi.nvmlInit()

deviceCount = nvidia_smi.nvmlDeviceGetCount()



while True:
   
    frame = stream.read()
    if frame is None:
        break
    
    start_time =time.time()
    combined_img = detect(session, frame, names)
    
    end_time=time.time()
    
    
    
    
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    
    gpu="Memory : ({:.2f}% free): {}(total) ".format( 100*info.free/info.total, info.total )
    gpu2="{} (free), {} (used)".format(info.free, info.used)
    fps=1/(end_time-start_time)
    
    cv2.putText(combined_img,gpu, (15,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    cv2.putText(combined_img,gpu2, (15,60), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    cv2.putText(combined_img,f"{fps:.3f} FPS " , (15,90), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    
    
    cv2.imshow("Output Frame", combined_img)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
nvidia_smi.nvmlShutdown()    
stream.stop()
cv2.destroyAllWindows()


