# This script trains the YOLOv8 model for acne detection.
# YOLO (You Only Look Once) is an object detection model — it draws
# bounding boxes around acne spots in face images.
# We start from a pre-trained YOLOv8 nano model and fine-tune it
# on our own labeled acne dataset.
import os
from ultralytics import YOLO

def train():
    # Detect the absolute path to the dataset's data.yaml
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'acne-detection-yolo.v1i.yolov8'))
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    
    # Loading a pre-trained YOLOv8 nano model — 'nano' means it's the
    # smallest and fastest variant, good for real-time detection.
    print("[*] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') 
    
    print(f"[*] Starting training using data from: {yaml_path}")
    print("[*] Training will utilize your RTX 3050 automatically.")
    
    # Training config:
    #   epochs=20: how many times to go through the entire dataset
    #   imgsz=640: the input image size YOLO expects
    #   batch=16: how many images to process at once
    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        workers=0,
        name='acne_detection_model'
    )
    
    print("\n[*] Training complete!")
    print(f"[*] Your trained model is saved in the 'runs/detect/acne_detection_model/weights/' folder.")
    print("[*] The best weights file is named 'best.pt'.")

if __name__ == '__main__':
    train()
