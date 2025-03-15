"""
Various custom data transforms to be used in datasets
"""
import torch
import torch.nn as nn 
from torchvision import transforms
import random
import numpy as np
from skimage.transform import resize
random.seed(0)
np.random.seed(0)

def RandomCropPair(x: torch.Tensor, y: torch.Tensor, size) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Crop two inputs with the same region. 
    Args:
        x: [N_1, H, W] 
        y: [N_2, H, W] 
        size: The size after cropping
    Returns:
        x_cropped, y_cropped, the cropped version of x and y where both are cropped to the same region
    """
    z = torch.cat([x,y]) 
    z_cropped = transforms.RandomCrop(size)(z) 
    return z_cropped[:x.shape[0]], z_cropped[x.shape[0]:] 
 
def PixelNormalize(images: torch.Tensor, gray_scale = False) -> torch.Tensor: 
    """
    Normalize RBG values of an image pixelwise based on the pixel values of other images. 
    RGB will be normalized seperately if gray_scale = False
    Args:
        images (Tensor): of shape [N, H, W] where it is assumed that 3 consecutative slices starting from 0 forms an RGB image. 
    Returns: 
        normalized_iimages (Tensor): [N, H, W] images but normalized
    """
    N, H, W = images.shape
    if gray_scale: 
        raise NotImplementedError("Currently only RBG is supported") 
    else:
        images = images.reshape(-1, 3, H, W) 
        norm_factor = torch.sqrt(images.pow(2).sum(dim=0, keepdim=True))
        normalized_images = images / norm_factor
        return normalized_images.reshape(-1, H, W)
    
    

"""
----------------------------------------------------------------------------------------------------------------------

"""

def arrayToTensor(array):
    if array is None:
        return array
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return tensor.float()

def rgbToGray(img):
    h, w, c = img.shape
    img = img[:,:,0] * 0.229 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114
    return img.reshape(h, w, 1)

def normalToMask(normal, thres=0.01):
    """
    Due to the numerical precision of uint8, [0, 0, 0] will save as [127, 127, 127] in gt normal,
    When we load the data and rescale normal by N / 255 * 2 - 1, the [127, 127, 127] becomes 
    [-0.003927, -0.003927, -0.003927]
    """
    mask = (np.square(normal).sum(2, keepdims=True) > thres).astype(np.float32)
    return mask

def randomCrop(inputs, target, size):
    if not __debug__: print('RandomCrop: input, target', inputs.shape, target.shape, size)
    h, w, _ = inputs.shape
    c_h, c_w = size
    if h == c_h and w == c_w:
        return inputs, target
    x1 = random.randint(0, w - c_w)
    y1 = random.randint(0, h - c_h)
    inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
    target = target[y1: y1 + c_h, x1: x1 + c_w]
    return inputs, target


def rescale(inputs, target, size):
    if not __debug__: print('Rescale: Input, target', inputs.shape, target.shape, size)
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        inputs = resize(inputs, size, order=1, mode='reflect')
        target = resize(target, size, order=1, mode='reflect')
    return inputs, target

def randomNoiseAug(inputs, noise_level=0.05):
    if not __debug__: print('RandomNoiseAug: input, noise level', inputs.shape, noise_level)
    noise = np.random.random(inputs.shape)
    noise = (noise - 0.5) * noise_level
    inputs += noise
    return inputs

def normalize(imgs):
    h, w, c = imgs[0].shape
    imgs = [img.reshape(-1, 1) for img in imgs]
    img = np.hstack(imgs)
    norm = np.sqrt((img * img).clip(0.0).sum(1))
    img = img / (norm.reshape(-1,1) + 1e-10)
    imgs = np.split(img, img.shape[1], axis=1)
    imgs = [img.reshape(h, w, -1) for img in imgs]
    return imgs
