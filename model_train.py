from ultralytics import YOLO

def main():
    # Load the model.
    model = YOLO('yolov8s.pt')

    # Training.
    results = model.train(
        data='data.yaml',
        imgsz=1080,
        epochs=50,
        batch=6,
        name='yolov8s_tanksmore'
    )

if __name__ == '__main__':
    main()
