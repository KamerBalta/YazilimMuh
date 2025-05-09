import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Paths for dataset
base_dir = r"C:\Users\nurka\OneDrive - gazi.edu.tr\Masaüstü\bonnDeneme\spektogramBonn"
split_dir = r"C:\Users\nurka\OneDrive - gazi.edu.tr\Masaüstü\bonnDeneme\splits"

# Ensure train and test directories exist
train_path = os.path.join(split_dir, 'train')
test_path = os.path.join(split_dir, 'test')

# Data augmentation and preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Initialize and customize ResNet50 with GAP
def initialize_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Global Average Pooling
    model.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
    
    # Modify final fully connected layers
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 3)  # 3 output classes
    )
    
    # Set all parameters to be trainable
    for param in model.parameters():
        param.requires_grad = True
    
    return model

# K-Fold Cross Validation with additional metrics
def k_fold_cross_validation_with_metrics(dataset_path, k=5, epochs=100, batch_size=32):
    # Load the full dataset for splitting
    full_dataset = datasets.ImageFolder(dataset_path, transform=data_transforms['train'])
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        print(f"Fold {fold + 1}/{k} starting...")

        # Split dataset into train and validation subsets
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model, criterion, and optimizer for each fold
        model = initialize_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

        best_accuracy = 0
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        all_labels, all_predictions = [], []

        for epoch in range(epochs):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            # Training loop
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = correct_train / total_train
            train_losses.append(running_loss / len(train_loader))
            train_accuracies.append(train_accuracy)

            # Validation loop
            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            fold_labels, fold_predictions = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    fold_labels.extend(labels.cpu().numpy())
                    fold_predictions.extend(predicted.cpu().numpy())

            val_accuracy = correct_val / total_val
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
            print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pth")

            # Store all labels and predictions for metric calculation
            all_labels.extend(fold_labels)
            all_predictions.extend(fold_predictions)

        # Compute metrics for the current fold
        fold_metrics.append({
            "accuracy": accuracy_score(all_labels, all_predictions),
            "precision": precision_score(all_labels, all_predictions, average="weighted"),
            "recall": recall_score(all_labels, all_predictions, average="weighted"),
            "f1_score": f1_score(all_labels, all_predictions, average="weighted"),
            "roc_auc": roc_auc_score(
                np.eye(3)[all_labels], 
                np.eye(3)[all_predictions], 
                multi_class="ovr"
            )
        })

        print(f"Metrics for Fold {fold + 1}: {fold_metrics[-1]}")

        # Plot metrics for this fold
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label="Train Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="red")
        plt.title(f"Loss During Training - Fold {fold + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_fold_{fold + 1}.png")
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(train_accuracies, label="Train Accuracy", color="blue")
        plt.plot(val_accuracies, label="Validation Accuracy", color="red")
        plt.title(f"Accuracy During Training - Fold {fold + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"accuracy_fold_{fold + 1}.png")
        plt.show()

    # Final metrics summary
    print(f"\nFinal Metrics: {fold_metrics}")
    avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0].keys()}
    print(f"Average Metrics: {avg_metrics}")

# Run K-Fold Cross Validation with metrics
k_fold_cross_validation_with_metrics(dataset_path=train_path, k=5, epochs=100, batch_size=32)
