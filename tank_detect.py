import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("C:/Users/MSI/Desktop/aman/yolo/runs/detect/yolov8m_tanks/weights/best.pt")

# IP camera URL
ip_camera_url = 'https://www.bing.com/videos/riverview/relatedvideo?&q=miltary+tanks&&mid=DE74892A013ACBF9F7B0DE74892A013ACBF9F7B0&mmscn=mtsc&aps=23&FORM=VRDGAR'

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

    # Perform inference with YOLOv8 model
    results = model(frame)

    # Process the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Assuming class 0 is 'tank', change if different
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = f'Tank {conf:.2f}'
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Tank Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
