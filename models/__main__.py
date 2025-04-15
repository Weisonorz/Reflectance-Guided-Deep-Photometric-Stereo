from . import PS_FCN_feature1
import torch
# from torchsummary import summary
from torchinfo import summary

device="cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters and input dimensions
batch_size = 10
height = 128
width = 128
image_channel = 3

# Random input data (batch of 4 images with 3 channels)
img_input = torch.rand(batch_size, image_channel*10, height, width)
light_input = torch.rand(batch_size, image_channel*10, height, width)
brdf = torch.rand(14,)

# Initialize the model
model = PS_FCN_feature1.PS_FCN_CBN(batch_size, fuse_type='max', batchNorm=True, c_in=image_channel).to(device)

# Model summary
summary(model, 
        input_data=(img_input.to(device), light_input.to(device), brdf.to(device)),  # pass as tuple
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=3)