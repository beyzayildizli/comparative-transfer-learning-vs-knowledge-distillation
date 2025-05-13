"""
@file: parameter_info.py
@description: Prints model summary, parameter count and FLOPs for ResNet18 and Light ResNet18
@assignment: Comparative Analysis of Transfer Learning and Knowledge Distillation in Deep Learning
@date: 12.05.2025
@authors: Beyza Yıldızlı @beyzayildizli10@gmail.com & Merve Öğ @merve.og@stu.fsm.edu.tr
"""

import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from train_light_model import ResNet18Light
from train_resnet18 import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18Light(num_classes=10).to(device)
#model = ResNet18(num_classes=10).to(device)

model.eval()

input_size = (1, 3, 32, 32)
dummy_input = torch.randn(input_size).to(device)

model_summary = summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "params_percent"])
print(model_summary)

dummy_input = torch.randn(1, 3, 224, 224)

flops = FlopCountAnalysis(model.cpu(), dummy_input)
params = parameter_count_table(model.cpu())

print(f"Toplam FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
print("Parametre Sayısı:\n", params)