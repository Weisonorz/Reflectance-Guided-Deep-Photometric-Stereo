import torch 
import torch.nn as nn 
from torch import Tensor
import math 

# class meanAngularError: 
#     def __init__(self): 
#         self.error: float = 0.
#         self.cnt: int = 0
        
#     def update(self, gt_n: Tensor, pred_n: Tensor, mask: Tensor | None = None) -> float: 
#         """
#         gt_n/pred_n: N x C x H x W
#         """
#         gt_n = gt_n / torch.linalg.norm(gt_n, dim=1, keepdim=True)
#         pred_n = pred_n / torch.linalg.norm(pred_n, dim=1, keepdim=True)
#         dot_product = (gt_n * pred_n).sum(dim=1).clamp(-1,1)
#         error_map   = torch.acos(dot_product) # [-pi, pi]
#         angular_map = error_map * 180.0 / math.pi
#         if mask is not None:
#             if mask.dtype != torch.bool:
#                 mask = mask.bool()
#             num_valid_elements = int(mask.sum().item())
#             if num_valid_elements > 0:
#                 masked_angular_map = angular_map[mask] # Apply mask directly
#                 sum_of_errors = masked_angular_map.sum()
#                 average_error = sum_of_errors / num_valid_elements
#             else:
#                 average_error = torch.tensor(0.0, dtype=gt_n.dtype, device=gt_n.device)
#                 sum_of_errors = torch.tensor(0.0, dtype=gt_n.dtype, device=gt_n.device)
#         else:
#             num_valid_elements = angular_map.numel()
#             sum_of_errors = angular_map.sum()
#             average_error = sum_of_errors / num_valid_elements

#         if self.cnt == 0:
#              self.error = average_error.item()
#         else:
#              self.error = ((self.error * self.cnt + sum_of_errors) / (self.cnt + num_valid_elements)).item() 

#         self.cnt += num_valid_elements # Keep cnt as a Python integer

#         return self.error  

class meanAngularError: 
    def __init__(self): 
        self.running_mean: float = 0.0
        self.total_cnt: int = 0
        
    def update(self, gt_n: Tensor, pred_n: Tensor, mask: Tensor | None = None) -> float: 
        """
        gt_n/pred_n: N x C x H x W
        """
        gt_n = gt_n / torch.linalg.norm(gt_n, dim=1, keepdim=True)
        pred_n = pred_n / torch.linalg.norm(pred_n, dim=1, keepdim=True)
        dot_product = (gt_n * pred_n).sum(dim=1).clamp(-1,1)
        error_map   = torch.acos(dot_product) # [-pi, pi]
        angular_map = error_map * 180.0 / math.pi

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            num_valid_elements = int(mask.sum().item())
            if num_valid_elements > 0:
                masked_angular_map = angular_map[mask] # Apply mask directly
                sum_of_errors = masked_angular_map.sum()
                average_error = sum_of_errors / num_valid_elements
            else:
                average_error = torch.tensor(0.0, dtype=gt_n.dtype, device=gt_n.device)
                num_valid_elements = 0  # Ensure we don’t update with zero elements
        else:
            num_valid_elements = angular_map.numel()
            sum_of_errors = angular_map.sum()
            average_error = sum_of_errors / num_valid_elements

        # Stable incremental mean update
        if num_valid_elements > 0:
            new_total_cnt = self.total_cnt + num_valid_elements
            self.running_mean += (sum_of_errors.item() - num_valid_elements * self.running_mean) / new_total_cnt
            self.total_cnt = new_total_cnt  # Update count

        return self.running_mean 