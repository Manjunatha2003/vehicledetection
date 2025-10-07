import cv2
import numpy as np
import torch
import time

# Initialize video capture
cap = cv2.VideoCapture('C:\\Users\\Manju\\Downloads\\vehicledetection\\vehicledetection\\tvid.mp4')

# Parameters
min_width_react = 80  # Minimum width of rectangle
min_height_react = 80  # Minimum height of rectangle
count_line_position = 550  # Position of the counting line

# Create background subtractor
algo = cv2.createBackgroundSubtractorMOG2()

# Function to get the center of the rectangle
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s model

# List to store the center points of detected objects
detect = []
offset = 6  # Allowable error between pixel
counter = 0

# Initialize SORT tracker
from sort import Sort
tracker = Sort()

while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    
    height, width, channels = frame1.shape

    # Apply YOLOv5
    results = model(frame1)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

    detections = []
    for i in range(len(labels)):
        x1, y1, x2, y2, conf = cord[i]
        if conf > 0.5:  # Confidence threshold
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)

            label = model.names[int(labels[i])]
            if label in ["car", "motorcycle", "bus", "truck"]:
                detections.append([x1, y1, x2, y2, conf])
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame1, (x1, y1 - 30), (x2, y1), (0, 0, 0), -1)
                cv2.putText(frame1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Update SORT tracker
    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))
        for x1, y1, x2, y2, obj_id in tracked_objects:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center = center_handle(x1, y1, x2-x1, y2-y1)
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)

            for (cx, cy) in detect:
                if cy < (count_line_position + offset) and cy > (count_line_position - offset):
                    counter += 1
                    detect.remove((cx, cy))

                # Draw bounding box and count
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame1, f"ID: {int(obj_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Apply background subtraction
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = algo.apply(blur)
    
    # Dilate the image
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the counting line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue
        
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)
        
        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                detect.remove((x, y))
        
        # Print and display the vehicle counter
        print("Vehicle Counter: " + str(counter))
        cv2.putText(frame1, "Vehicle Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    cv2.imshow('video original', frame1)
    
    if cv2.waitKey(1) == 13:  # Enter key to exit
        break

cv2.destroyAllWindows()
cap.release()
