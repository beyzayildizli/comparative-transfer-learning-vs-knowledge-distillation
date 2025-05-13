"""
@file: train_light_model.py
@description: Lightweight ResNet18 model training on CIFAR-10 to enable comparison with post-distillation performance.
@assignment: Comparative Analysis of Transfer Learning and Knowledge Distillation in Deep Learning
@date: 08.05.2025
@authors: Beyza Yıldızlı @beyzayildizli10@gmail.com & Merve Öğ @merve.og@stu.fsm.edu.tr
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18Light(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Light, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train_and_save_best_model(model, train_loader, test_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_accuracy = 0.0
    best_model_state_dict = None
    best_metrics = None
    best_preds = None
    best_labels = None
    best_epoch = 0  

    os.makedirs("knowledge-distillation", exist_ok=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        accuracy, precision, recall, f1, preds, labels = test(model, test_loader, device, epoch + 1, epochs)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state_dict = model.state_dict()
            best_metrics = (accuracy, precision, recall, f1)
            best_preds = preds
            best_labels = labels
            best_epoch = epoch + 1 
            print(f"New best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")

    if best_model_state_dict is not None:
        model_save_path = "knowledge-distillation/saved_models/lightmodel_64_adam.pth"
        torch.save(best_model_state_dict, model_save_path)
        print(f"\nBest model saved from epoch {best_epoch} with accuracy: {best_accuracy:.2f}%")

        acc, prec, rec, f1 = best_metrics
        print("\n--- Best Model Metrics ---")
        print(f"Test Accuracy: {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall: {rec*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")

        cm = confusion_matrix(best_labels, best_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)

        metrics_text = f"Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1 Score: {f1*100:.2f}%"
        plt.title(f"Best Epoch ({best_epoch}/{epochs}) Confusion Matrix\n{metrics_text}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        os.makedirs("knowledge-distillation/results", exist_ok=True)
        save_path = f"knowledge-distillation/results/lightmodel_64_adam.png"
        plt.savefig(save_path)
        plt.close()

        print(f"Confusion matrix for best epoch saved as {save_path}")

def test(model, test_loader, device, epoch, last_epoch):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1, all_preds, all_labels


if __name__ == "__main__":
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    epochs = 200
    teacher_model = ResNet18Light(num_classes=10).to(device)
    train_and_save_best_model(teacher_model, train_loader, test_loader, epochs=epochs, learning_rate=0.001, device=device)
