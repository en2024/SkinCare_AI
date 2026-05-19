"""
Training script for the YOLOv8 acne detection model.

Fine-tunes YOLOv8 nano on a custom labeled acne dataset to detect
and localize acne spots in face images.

Usage:  python scripts/train_yolo.py
"""
import os
from ultralytics import YOLO


def train():
    # Path to the dataset config file
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'acne-detection-yolo.v1i.yolov8'))
    yaml_path = os.path.join(dataset_dir, 'data.yaml')

    # Start from a pretrained YOLOv8 nano model (small and fast)
    print("[*] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    print(f"[*] Starting training using data from: {yaml_path}")

    # epochs=20: full passes through the dataset
    # imgsz=640: input image resolution
    # batch=16: images processed at once
    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        workers=0,
        name='acne_detection_model'
    )

    print("\n[*] Training complete!")
    print("[*] Best weights saved in 'runs/detect/acne_detection_model/weights/best.pt'.")

if __name__ == '__main__':
    train()
