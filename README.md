[App](https://kilndetector.herokuapp.com/)

Creating a streamlit application for YOLOv4 custom trained image detection model.

darknet_app.py is somewhat a copy, and credits are *inside* darknet_app.py=
[Actual Code](https://medium.com/dev-genius/a-simple-object-detection-app-built-using-streamlit-and-opencv-4365c90f293c)

Blog About Heroku Deployment=
[Link](https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku/)

![APP](https://github.com/Aryaman-Chauhan/YOLOv4_Heroku_Deployment/blob/main/data/st1.jpg)

![Detecting](https://github.com/Aryaman-Chauhan/YOLOv4_Heroku_Deployment/blob/main/data/st2.jpg)


Later, the same data was trained on YOLOv7, and was later exported to .onnx. 

This model was more effective in detection tasks and outperformed the v4 model.

app.py uses this exported .onnx weighs to detect images, credits = 
[By Wong Kin Yiu](https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb)

The app has similar functionality accept for the inability to change confidence and nms threshold values.
