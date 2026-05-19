"""
Training script for the ResNet50 skin type classifier.

Fine-tunes a pretrained ResNet50 on the 5-class skin type dataset
(oily, dry, normal, combination, sensitive) and saves the best weights.

Usage:  python scripts/train_skin_model.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import copy


def train_model():
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")

    # ── Data augmentation and preprocessing ──────────────────────────────
    # Training images are randomly cropped and flipped for augmentation.
    # Validation images are simply resized and center-cropped.

    data_transforms = {
        'Train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Skin Type Identification Research'))

    # ── Load datasets from folder structure ──────────────────────────────
    # Expects: data_dir/Train/{class_name}/*.jpg
    #          data_dir/Validation/{class_name}/*.jpg

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['Train', 'Validation']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
        for x in ['Train', 'Validation']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation']}
    class_names = image_datasets['Train'].classes

    print(f"[*] Found {dataset_sizes['Train']} train, {dataset_sizes['Validation']} val.")
    print(f"[*] Classes: {class_names}")

    num_classes = len(class_names)

    # ── Model setup ──────────────────────────────────────────────────────
    # Start with a pretrained ResNet50 and replace the final layer
    # to output 5 classes instead of ImageNet's 1000.

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # ── Training loop ────────────────────────────────────────────────────

    print("[*] Starting training...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Only compute gradients during training, not validation
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Keep the weights from the epoch with the best validation accuracy
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    print(f'[*] Best val acc: {best_acc:4f}')

    # ── Save the best model ──────────────────────────────────────────────

    model.load_state_dict(best_model_wts)

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analyzer', 'ml_models', 'model_resnet50_5class.pth'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(model.state_dict(), output_path)
    print(f'[*] Saved model: {output_path}')

if __name__ == '__main__':
    train_model()
