from sort import *
import cv2, numpy as np, pandas as pd
from Yolov8.yolov8 import YOLOv8_det  # Import YOLOv8 object detector
import time, pytz, os
from datetime import datetime

# Set of class IDs for vehicles
vehicles = set([2, 3, 5, 7])  # Update these class IDs as needed

# Set up timezone
tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Load video
cap = cv2.VideoCapture('./test_image/challenge.mp4')
video_out_path = './test_image/out_challenge.mp4'

# Check if video capture is opened successfully
if not cap.isOpened():
    print("Error: Unable to open input video.")
    exit()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS),
                          (1020, 500))

# Check if video writer is initialized successfully
if not cap_out.isOpened():
    print("Error: Unable to open output video.")
    exit()

# Object tracker
mot_tracker = Sort()

# Model paths
model_path_object = "./models/yolov8n.onnx"

# Initialize YOLOv8 object detector
my_yolov8 = YOLOv8_det(model_path_object)

# Dictionary to store colors for each vehicle ID
id_colors = {}

# Function to generate random colors
def get_random_color():
    return tuple(np.random.randint(0, 255, size=3).tolist())

# Dictionary to store the path (history of positions) for each vehicle
vehicle_paths = {}

down = {}
up = {}
counter_down = []
counter_up = []

# Position of lines for vehicle counting
red_line_y = 163
blue_line_y = 248
offset = 8


if __name__ == "__main__":
    print("Starting the application")
    frame_nmr = -1
    car_count = 0
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if not ret:
            break  # Stop if no more frames are available

        frame = cv2.resize(frame, (1020, 500))
        detections = my_yolov8.detect_objects(frame)
        boxes, scores, class_ids = detections
        detections_ = []
        frame = my_yolov8.draw_detections(frame)

        for i in range(len(boxes)):
            box = boxes[i].tolist()
            score = scores[i].tolist()
            class_id = class_ids[i].tolist()
            if int(class_id) in vehicles:
                detections_.append(box + [score, class_id])

        # Create dataframe of detections
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_))

            for track_id in track_ids:
                x3, y3, x4, y4, id = map(int, track_id)
                cx = int((x3 + x4) // 2)
                cy = int((y3 + y4) // 2)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
                cv2.putText(frame, str(int(id)), (x3, y3-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # Assign color to vehicle if not already assigned
                if id not in id_colors:
                    id_colors[id] = get_random_color()

                color = id_colors[id]

                # Add the current position to the vehicle's path
                if id not in vehicle_paths:
                    vehicle_paths[id] = []
                vehicle_paths[id].append((cx, y4))

                # Keep only the last 50 positions to avoid too long of a trail
                if len(vehicle_paths[id]) > 50:
                    vehicle_paths[id] = vehicle_paths[id][-50:]

                # Draw the path of the vehicle with the same color
                for j in range(1, len(vehicle_paths[id])):
                    pt1 = vehicle_paths[id][j - 1]
                    pt2 = vehicle_paths[id][j]
                    cv2.line(frame, pt1, pt2, color, 2, -1)

        cv2.imshow('Vehicle Detection', frame)
        cap_out.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()
