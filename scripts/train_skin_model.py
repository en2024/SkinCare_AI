import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import copy

def train_model():
    # Detect GPU hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training will run on: {device}")

    # Define Image Augmentations/Transformations
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
    
    # Load dataset
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
    
    print(f"[*] Found {dataset_sizes['Train']} training images and {dataset_sizes['Validation']} validation images.")
    print(f"[*] Classes: {class_names}")

    num_classes = len(class_names)

    # Load pre-trained ResNet-50 architecture
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace final fully connected layer for the new 5 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("[*] Beginning training using CUDA via RTX 3050...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model weights if better
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'[*] Training complete. Best Validation Accuracy: {best_acc:4f}')

    # Save the best weights
    model.load_state_dict(best_model_wts)
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analyzer', 'ml_models', 'model_resnet50_5class.pth'))
    # ensure dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(model.state_dict(), output_path)
    print(f'[*] Saved retrained model to: {output_path}')

if __name__ == '__main__':
    train_model()
