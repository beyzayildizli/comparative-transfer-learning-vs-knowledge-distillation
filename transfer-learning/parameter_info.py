"""
@file: parameter_info_transfer.py
@description: Prints model summary, parameter count and FLOPs for transfer learning model with partially frozen ResNet18
@assignment: Comparative Analysis of Transfer Learning and Knowledge Distillation in Deep Learning
@date: 12.05.2025
@authors: Beyza Yıldızlı @beyzayildizli10@gmail.com & Merve Öğ @merve.og@stu.fsm.edu.tr
"""

import torch
from torch import nn
from torchinfo import summary
from torchvision import models
from fvcore.nn import FlopCountAnalysis, parameter_count_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 10)
)

model = model.to(device)
model.eval()

input_size = (1, 3, 32, 32)
dummy_input = torch.randn(input_size).to(device)

model_summary = summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "params_percent"])
print(model_summary)

flops = FlopCountAnalysis(model.cpu(), dummy_input.cpu())
params = parameter_count_table(model.cpu())

print(f"\nToplam FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
print("\nParametre Sayısı:\n", params)
