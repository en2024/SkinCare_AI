import os
from ultralytics import YOLO

def train():
    # Detect the absolute path to the dataset's data.yaml
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'acne-detection-yolo.v1i.yolov8'))
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    
    # We load a pre-trained YOLOv8 nano model (fastest and lightest)
    print("[*] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') 
    
    print(f"[*] Starting training using data from: {yaml_path}")
    print("[*] Training will utilize your RTX 3050 automatically.")
    
    # Train the model
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
