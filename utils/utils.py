import torch 
import numpy as np
from torch import Tensor

def lightpos2angle(positions) -> Tensor: 
    """
    Given a light position, returns unit vector from that position to (0,0,0). 
    Args: 
        positions (ArrayLike): [N, 3] or [3]
    Returns:
        angle (Tensor): [N,3] or [3]
    """
    if not isinstance(positions, torch.Tensor): 
        positions = torch.Tensor(positions)
        
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)  # Add a batch dimension if it's a single point

    origin = torch.zeros_like(positions)  # Create a tensor of zeros with the same shape as positions
    direction_vectors = origin - positions  # Calculate the vector from the light position to the origin

    # Normalize the direction vectors to get unit vectors
    magnitudes = torch.linalg.vector_norm(direction_vectors, dim=-1, keepdim=True)
    unit_vectors = direction_vectors / magnitudes

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
    Given a normal map of shape [N, 3, H, W] or [3, H, W], get the mask that contains all valid normals
    """
    return torch.linalg.norm(normals, dim=-3) > 1e-5  