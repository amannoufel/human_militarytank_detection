import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open a connection to the IP camera
ip_camera_url = 'http://192.168.88.189:81/stream'
cap = cv2.VideoCapture(ip_camera_url)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Extract bounding boxes and labels
    person_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Class ID for 'person' in COCO dataset
                person_boxes.append(box)

    # Draw bounding boxes on the frame
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the number of people detected
    num_people = len(person_boxes)
    cv2.putText(frame, f'People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Inference", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the IP camera and close the display window
cap.release()
cv2.destroyAllWindows()
