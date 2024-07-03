import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model for tanks
tank_model = YOLO("C:/Users/MSI/Desktop/aman/yolo/runs/detect/yolov8m_tanks/weights/best.pt")

# Load the YOLOv8 model for people
person_model = YOLO("yolov8n.pt")

# IP camera URL
ip_camera_url = 'C:/Users/MSI/Downloads/cutteryt.mp4'

# Open a connection to the IP camera
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open IP camera stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Perform inference with YOLOv8 model for tanks
    tank_results = tank_model(frame)

    # Perform inference with YOLOv8 model for people
    person_results = person_model(frame)

    # Process the results for tanks
    for tank_result in tank_results:
        tank_boxes = tank_result.boxes
        for box in tank_boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Assuming class 0 is 'tank', change if different
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = f'Tank {conf:.2f}'
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Process the results for people
    person_boxes = []
    for person_result in person_results:
        for box in person_result.boxes:
            if box.cls == 0:  # Class ID for 'person' in COCO dataset
                person_boxes.append(box)

    # Draw bounding boxes for people on the frame
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the number of people detected
    num_people = len(person_boxes)
    cv2.putText(frame, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
