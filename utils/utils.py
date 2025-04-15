import torch 
import numpy as np
from torch import Tensor

def lightpos2angle(positions) -> Tensor: 
    """
    Convert raw light positions to normailzed angles with the same coordinate system as in PS-FCN. 
    Args: 
        positions (ArrayLike): [N, 3] or [3]
    Returns:
        angle (Tensor): [N,3] or [3]
    """
    if not isinstance(positions, torch.Tensor): 
        positions = torch.Tensor(positions)
        
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)  # Add a batch dimension if it's a single point

    positions[..., 2] = -positions[..., 2]  # Invert the z-coordinate
    # Normalize the direction vectors to get unit vectors
    magnitudes = torch.linalg.vector_norm(positions, dim=-1, keepdim=True)
    unit_vectors = positions / magnitudes

    if unit_vectors.shape[0] == 1 and len(unit_vectors.shape)>1:
        return unit_vectors.squeeze(0)
    return unit_vectors


def broadcast_angles(angles: Tensor, height, width) -> Tensor: 
    """
    Args:
        angles: of shape [N, num_lights, 3] or [num_lights, 3]
    
    Returns: 
        broadcasted_angles: of shape [N, 3*num_lights, height, width] or [3*num_lights, height, width]
    """
    broadcasted_angles = angles.unsqueeze(-1).unsqueeze(-1)
    if angles.ndim == 2: 
        broadcasted_angles = broadcasted_angles.expand(angles.shape[0], 3, height, width)
        broadcasted_angles = broadcasted_angles.view(-1, height, width)
    else: 
        broadcasted_angles = broadcasted_angles.expand(angles.shape[0], angles.shape[1], 3, height, width)
        broadcasted_angles = broadcasted_angles.view(angles.shape[0], -1, height, width)
    
    return broadcasted_angles


def get_mask(normals: Tensor): 
    """
    Given a normal map of shape [N, 3, H, W] or [3, H, W], get the mask that contains all valid normals (i.e the forground)
    """
    return torch.linalg.norm(normals, dim=-3) > 1e-5  

import os

def makeFile(f):
    if not os.path.exists(f):
        os.makedirs(f)

def makeFiles(f_list):
    for f in f_list:
        makeFile(f)

def dictToString(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def checkIfInList(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains