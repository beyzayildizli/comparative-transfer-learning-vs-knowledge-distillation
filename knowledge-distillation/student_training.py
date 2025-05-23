"""
@file: student_training.py
@description: Training a student model using knowledge distillation from a pretrained teacher model (ResNet18).
@assignment: Comparative Analysis of Transfer Learning and Knowledge Distillation in Deep Learning
@date: 08.05.2025
@authors: Beyza Yıldızlı @beyzayildizli10@gmail.com & Merve Öğ @merve.og@stu.fsm.edu.tr
"""

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    from train_resnet18 import ResNet18
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from train_light_model import ResNet18Light
    import random
    import numpy as np

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        optimizer = optim.Adam(student.parameters(), lr=learning_rate)
        #optimizer = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

        best_accuracy = 0.0
        best_model_state_dict = None
        best_metrics = None  
        best_preds = None
        best_labels = None
        best_epoch = 0

        os.makedirs("knowledge-distillation", exist_ok=True)

        teacher.eval()
        student.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                student_logits = student(inputs)

                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
                label_loss = ce_loss(student_logits, labels)
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            accuracy, precision, recall, f1, preds, labels = test(student, test_loader, device)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.2f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state_dict = student.state_dict()
                best_metrics = (accuracy, precision, recall, f1)
                best_preds = preds
                best_labels = labels
                best_epoch = epoch + 1
                print(f"New best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")

        if best_model_state_dict is not None:
            model_save_path = "knowledge-distillation/saved_models/kd_student_64_adam_T2.pth"
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
            save_path = f"knowledge-distillation/results/kd_student_64_adam_T2.png"
            plt.savefig(save_path)
            plt.close()

            print(f"Confusion matrix for best epoch saved as {save_path}")

    def test(model, test_loader, device):
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

    epochs = 200

    teacher_model_path = "knowledge-distillation/saved_models/resnet18_64_adam.pth"
    if not os.path.exists(teacher_model_path):
        print("Öğretmen modeli bulunamadı! Önce modeli eğitin.")
        exit()
    
    teacher_model = ResNet18(num_classes=10).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path))
    teacher_model.eval()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    student_model = ResNet18Light(num_classes=10).to(device)
    #student_model = ResNet18(num_classes=10).to(device)

    train_knowledge_distillation(teacher=teacher_model, student=student_model, train_loader=train_loader, epochs=epochs, learning_rate=0.001, T=2, soft_target_loss_weight=0.5, ce_loss_weight=0.5, device=device)