"""
Created on Thu Jul 21 17:11:58 2022
@url: https://blog.devgenius.io/a-simple-object-detection-app-built-using-streamlit-and-opencv-4365c90f293c
@author: Bolarinwa Oreoluwa
@Edited-and-customized-by: Aryaman Chauhan
"""
import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import time, requests, random
import onnxruntime as ort
from pathlib import Path
from collections import OrderedDict,namedtuple

#import tensorflow as tf
#import tensorflow_hub as hub
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
def object_detection_image():
	st.title('Brick Kiln Detector')
	st.subheader("""
	This object detection project takes in an image and outputs the image with bounding boxes created around the brick kilns in the image.
	""")
	cuda = False
	file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg', 'tiff', 'tif', 'svg'])
	if file!= None:
		img1 = Image.open(file).convert('RGB')
		
		if st.checkbox("Enhance Sharpness"):
			curr_sharp = ImageEnhance.Sharpness(img1)
			sharp = st.slider('Sharpness', 0.0, 2.0, 1.0, 0.1)
			img1 = curr_sharp.enhance(sharp)
			
		if st.checkbox("Enhance Contrast"):
			curr_cont = ImageEnhance.Contrast(img1)
			cont = st.slider('Contrast', 0.0, 2.0, 1.0, 0.1)
			img1 = curr_cont.enhance(cont)
		
		if st.checkbox("Enhance Color"):
			curr_col = ImageEnhance.Color(img1)
			col = st.slider('Color', 0.0, 2.0, 1.0, 0.1)
			img1 = curr_col.enhance(col)
		
		if st.checkbox("Enhance Brightness"):
			curr_bright = ImageEnhance.Brightness(img1)
			bright = st.slider('Brightness', 0.0, 2.0, 1.0, 0.1)
			img1 = curr_bright.enhance(bright)
		
		img2 = np.array(img1)
		BASE_PATH = os.path.dirname(os.path.abspath(__file__))
		weights_path = os.path.sep.join([BASE_PATH, "epoch_1724.onnx"])
		providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
		session = ort.InferenceSession(weights_path, providers=providers)

		st.image(img2, caption = "Uploaded Image")
		my_bar = st.progress(0)
		#classNames = []
		classNames = ['brick_kiln']
		colors = {name:[255, 0, 0] for i,name in enumerate(classNames)}
		#f = open(r'C:\Users\Olazaah\Downloads\stream\labels\coconames.txt','r')
		#lines = f.readlines()
		#classNames = [line.strip() for line in lines]
		image = img2.copy()
		image, ratio, dwdh = letterbox(image, auto=False)
		image = image.transpose((2, 0, 1))
		image = np.expand_dims(image, 0)
		image = np.ascontiguousarray(image)

		im = image.astype(np.float32)
		im /= 255
		
		outname = [i.name for i in session.get_outputs()] 
		inname = [i.name for i in session.get_inputs()]
		inp = {inname[0]:im}
		outputs = session.run(outname, inp)[0]
		
		obj_list=[]
		confi_list =[]
		df = None

		ori_images = [img2.copy()]

		for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
			image = ori_images[int(batch_id)]
			obj_list.append(classNames[int(cls_id)].upper())
			confi_list.append(int(score*100))			
			box = np.array([x0,y0,x1,y1])
			box -= np.array(dwdh*2)
			box /= ratio
			box = box.round().astype(np.int32).tolist()
			cls_id = int(cls_id)
			score = round(float(score),3)
			name = classNames[cls_id]
			color = colors[name]
			name += ' '+str(score * 100)[:4]
			cv2.rectangle(image,box[:2],box[2:],color,2)
			cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0, 0, 255],thickness=4)
	
		df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
		if st.checkbox("Show Object's list" ):
			st.write(df)

		if st.checkbox("Show Confidence bar chart" ):
			st.subheader('Bar chart for confidence levels')
			st.bar_chart(df["Confidence"])
		st.image(ori_images[0], caption='Proccesed Image')
		
		cv2.waitKey(0)
		
		cv2.destroyAllWindows()
		my_bar.progress(100)

def main():
	st.title("IIRS, Dehradun")
	new_title = '<p style="font-size: 42px;">Welcome to Brick Kiln Detection App!</p>'
	read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

	read_me = st.markdown("""
	This project was built using Streamlit and OpenCV 
	to demonstrate YOLO Object detection in images.
	
	To use the application, go to Detector in Sidebar on the Left.

	""")
	st.sidebar.title("Select Activity")
	choice  = st.sidebar.selectbox("MODE",("Kiln Detector","About"))
	#["Show Instruction","Landmark identification","Show the #source code", "About"]
	
	if choice == "Kiln Detector":
		#st.subheader("Object Detection")
		read_me_0.empty()
		read_me.empty()
		#st.title('Object Detection')
		object_detection_image()

	elif choice == "About":
		print()
		

if __name__ == '__main__':
		main()	
