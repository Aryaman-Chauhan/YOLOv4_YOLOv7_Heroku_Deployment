import cv2
import numpy as np
import os

BASE_PATH = os.getcwd()
cfg_file = os.path.sep.join([BASE_PATH, "yolov4-obj-test.cfg"])
weights_file = os.path.sep.join([BASE_PATH, "custom.weights"])

net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
classes = ['brick_kiln']

img = cv2.imread(os.path.sep.join([BASE_PATH, "2.png"]))
font = cv2.FONT_HERSHEY_PLAIN

height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)


for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i],2))
    color = (0, 0, 255)
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

cv2.imshow("Image", img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()