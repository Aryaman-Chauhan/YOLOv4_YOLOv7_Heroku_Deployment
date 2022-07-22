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
#import tensorflow as tf
#import tensorflow_hub as hub

def object_detection_image():
    st.title('Brick Kiln Detector')
    st.subheader("""
    This object detection project takes in an image and outputs the image with bounding boxes created around the brick kilns in the image.
    """)
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
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

        st.image(img1, caption = "Uploaded Image")
        my_bar = st.progress(0)
        confThreshold =st.slider('Confidence', 0, 100, 20)
        nmsThreshold= st.slider('Threshold', 0, 100, 20)
        #classNames = []
        whT = 320
        classNames = ['brick_kiln']
        #f = open(r'C:\Users\Olazaah\Downloads\stream\labels\coconames.txt','r')
        #lines = f.readlines()
        #classNames = [line.strip() for line in lines]
        BASE_PATH = os.getcwd()
        config_path = os.path.sep.join([BASE_PATH, "yolov4-obj-test.cfg"])
        weights_path = os.path.sep.join([BASE_PATH, "custom.weights"])
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs,img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold/100):
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        
            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, nmsThreshold/100)
            obj_list=[]
            confi_list =[]
            #drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(img2, (x, y), (x+w,y+h), (0, 0 , 255), 2)
                #print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())
                
                confi_list.append(int(confs[i]*100))
                cv2.putText(img2,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
            if st.checkbox("Show Object's list" ):
                
                st.write(df)
            if st.checkbox("Show Confidence bar chart" ):
                st.subheader('Bar chart for confidence levels')
                
                st.bar_chart(df["Confidence"])
           
        blob = cv2.dnn.blobFromImage(img2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,img2)
    
        st.image(img2, caption='Proccesed Image')
        
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
    choice  = st.sidebar.selectbox("MODE",("About","Kiln Detector"))
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
