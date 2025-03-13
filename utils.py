import torch 
from torch import Tensor

def lightpos2angle(positions: Tensor) -> Tensor: 
    """
    Given a light position, returns unit vector from that position to (0,0,0). 
    Args: 
        positions (Tensor): [N, 3] or [3]
    Returns:
        angle (Tensor): [N,3] or [3]
    """
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



