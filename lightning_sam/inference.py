import torch

from model import Model

ckpt_path = "../out/training_merged_scenario/"
ckpt_name = "epoch-000025-f10.94-ckpt"

model = Model()
checkpoint = torch.load(ckpt_path + ckpt_name + ".pth")
model.load_state_dict(checkpoint['model_state_dict'])