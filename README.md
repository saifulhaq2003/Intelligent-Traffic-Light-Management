# Intelligent-Traffic-Light-Management
Computer Vision driven Deep Learning implementation.

Here, we firstly use Computer Vision to use process each video frame-wise. We then apply Transfer Learning's YOLOv8 model to detect objects from its COCO dataset. We filter out only the cars, buses, trucks and motorcycles and log them in a .csv file along with a calculated traffic light duration through a research-based formula. 

Our dataset is prepared.

Now we apply, Bi-directional Long Short Term Memory(B-LSTM) to predict the traffic light durations for each time stamps based on the vehicle count pattern.
