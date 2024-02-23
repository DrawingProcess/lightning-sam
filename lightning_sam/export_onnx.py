from torch.autograd import Variable
import torch.onnx
import torchvision
import torch

from model import Model

ckpt_path = "../out/training_merged_scenario/"
ckpt_name = "epoch-000025-f10.94-ckpt"

model = Model()
dummy_input = Variable(torch.randn(1, 3, 256, 256))
state_dict = torch.load(ckpt_path + ckpt_name + ".pth")
model.load_state_dict(state_dict['model_state_dict'])
torch.onnx.export(model, dummy_input, ckpt_path + ckpt_name + ".onnx")
