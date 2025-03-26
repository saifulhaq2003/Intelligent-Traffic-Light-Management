import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import numpy as np
import csv
import time

model=YOLO('yolov8s.pt')

class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

tracker=Tracker()
cap=cv2.VideoCapture('traffic_video.avi')
down={}
up={}

counter_down=[]
counter_up=[]

# ✅ CSV File Setup
csv_filename = "vehicles.csv"
# ✅ Create CSV file with headers if not exists
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Vehicle ID", "Vehicle Type", "Timestamp", "Red Line Y", "Blue Line Y"])

# ✅ Store vehicle IDs that crossed the red line first
crossed_red = {}

# ✅ Store vehicle IDs that completed crossing both lines
counter_down = []

# ✅ Define red and blue lines
red_line_y = 278
blue_line_y = 315
offset = 7  # Allow slight variation due to tracking jitter

# ✅ Define vehicle classes to track
vehicle_classes = ["car", "bus", "truck", "motorcycle"]  
vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}  # ✅ Count of each type

while True:    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # YOLO Detection
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    # ✅ Store detections
    detections = []
    detected_vehicles = {}  # ✅ Store detected vehicles and their types

    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        vehicle_type = class_list[d]  # Get detected class

        if vehicle_type in vehicle_classes:
            detections.append([x1, y1, x2, y2])
            detected_vehicles[(x1, y1, x2, y2)] = vehicle_type  # Store type with coordinates

    # ✅ Ensure 'detections' is not empty
    bbox_id = tracker.update(np.array(detections)) if detections else []

    # ✅ Process ALL detected vehicles
    for bbox in bbox_id:
        x3, y3, x4, y4, vehicle_id = bbox  # Get bounding box & ID
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2  # Compute center
        
        # ✅ Get vehicle type from stored detections
        vehicle_type = detected_vehicles.get((x3, y3, x4, y4), "unknown")

        # ✅ Check if the vehicle crosses the RED line first
        if red_line_y - offset < cy < red_line_y + offset:
            crossed_red[vehicle_id] = (cy, vehicle_type)  # Store ID & Type

        # ✅ Only consider vehicles that crossed red before reaching blue
        if vehicle_id in crossed_red and blue_line_y - offset < cy < blue_line_y + offset:
            _, v_type = crossed_red[vehicle_id]  # Get stored vehicle type

            # ✅ Draw circle at the blue line for valid vehicles
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{v_type}-{vehicle_id}", (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # ✅ Add to counter only once
            if vehicle_id not in counter_down:
                counter_down.append(vehicle_id)
                vehicle_counts[v_type] += 1  # ✅ Increment count of detected type

                # ✅ Save data to CSV
                with open(csv_filename, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([vehicle_id, v_type, time.strftime("%Y-%m-%d %H:%M:%S"), crossed_red[vehicle_id][0], cy])

    # ✅ Draw reference lines
    cv2.line(frame, (260, 278), (724, 278), (0, 0, 255), 3)  # Red line
    cv2.putText(frame, 'red line', (260, 278), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.line(frame, (200, 320), (779, 315), (255, 0, 0), 3)  # Blue line
    cv2.putText(frame, 'blue line', (200, 315), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ✅ Display vehicle count
    cv2.putText(frame, f"Cars: {vehicle_counts['car']}  Buses: {vehicle_counts['bus']}  Trucks: {vehicle_counts['truck']}  Motorcycles: {vehicle_counts['motorcycle']}", 
                (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)    

    cv2.imshow("frames", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
