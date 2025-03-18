from . import PS_FCN_feature1
import torch
from torchsummary import summary

device="cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters and input dimensions
batch_size = 4
height = 128
width = 128
c_in_new = 3

# Random input data (batch of 4 images with 3 channels)
img_input = torch.rand(batch_size, c_in_new, height, width)

# Random lighting input (optional)
light_input = torch.rand(batch_size, c_in_new, height, width)

brdf = torch.rand(14,)

# Initialize the model
model = PS_FCN_feature1.PS_FCN_CBN(batch_size, height, width, fuse_type='max', batchNorm=True, c_in=6).to(device)

# Model summary
# summary(model, [(c_in_new, height, width), (c_in_new, height, width), (14,)])
output = model([img_input.to(device), light_input.to(device), brdf.to(device)])
print(output.shape)
print(output)