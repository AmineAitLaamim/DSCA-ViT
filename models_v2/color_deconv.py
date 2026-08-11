import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class ColorDeconvolution(nn.Module):
    """
    Color Deconvolution Module using the Ruifrok & Johnston method.
    
    This module separates RGB Immunohistochemistry (IHC) images into Hematoxylin 
    and DAB channels by converting the images to Optical Density (OD) space and 
    applying the inverse of the H-DAB stain matrix.
    
    This is a fixed (non-learnable) preprocessing step based on the Beer-Lambert law.
    """
    
    def __init__(self, epsilon: float = 1e-6) -> None:
        """
        Initializes the ColorDeconvolution module with the fixed H-DAB stain matrix.
        
        Args:
            epsilon (float, optional): A small value to avoid log(0) when converting
                                       to Optical Density space. Defaults to 1e-6.
        """
        super().__init__()
        self.epsilon = epsilon
        
        # Ruifrok H-DAB stain matrix (rows are stain vectors in RGB OD space)
        # H: Hematoxylin, DAB: 3,3'-Diaminobenzidine, Res: Residual
        stain_matrix = torch.tensor([
            [0.6500286, 0.7040310, 0.2860126],
            [0.2688606, 0.5700937, 0.7767574],
            [0.7110272, 0.4234194, 0.5615672]
        ], dtype=torch.float32)
        
        # Calculate the inverse of the stain matrix
        stain_matrix_inv = torch.linalg.inv(stain_matrix)
        
        # Register as a buffer so it is automatically moved to the correct device
        # (CPU/GPU) but not updated during backpropagation (not trainable).
        self.register_buffer("stain_matrix_inv", stain_matrix_inv)

    def forward(self, x_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies color deconvolution to the input RGB images.
        
        Args:
            x_rgb (torch.Tensor): Input RGB images of shape (B, 3, H, W). 
                                  Values should be unnormalized RGB tensors 
                                  in the range [0, 1].
                                  
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the Hematoxylin 
                                               and DAB channels, each of shape 
                                               (B, 1, H, W) and clamped to be >= 0.
        """
        # Convert RGB to Optical Density (OD): OD = -log10(I / I0)
        # Assuming I0 = 1 for normalized images in [0, 1].
        od = -torch.log10(x_rgb + self.epsilon)
        
        # Rearrange OD tensor for matrix multiplication: (B, 3, H, W) -> (B, H, W, 3)
        od_reshaped = od.permute(0, 2, 3, 1)
        
        # Apply the inverse stain matrix to get the stain concentrations
        # OD (..., 3) @ M_inv (3, 3) -> Stains (..., 3)
        stains = torch.matmul(od_reshaped, self.stain_matrix_inv)
        
        # Rearrange back to channels first: (B, H, W, 3) -> (B, 3, H, W)
        stains = stains.permute(0, 3, 1, 2)
        
        # Extract H and DAB channels (indices 0 and 1)
        h_channel = stains[:, 0:1, :, :]
        dab_channel = stains[:, 1:2, :, :]
        
        # Clamp negative values to 0 (negative OD values are non-physical)
        h_channel = torch.clamp(h_channel, min=0.0)
        dab_channel = torch.clamp(dab_channel, min=0.0)
        
        return h_channel, dab_channel


def deconvolve_numpy(image_rgb_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for color deconvolution on a single numpy image.
    Useful for visualization and debugging purposes.
    
    Args:
        image_rgb_uint8 (np.ndarray): Input RGB image of shape (H, W, 3) with 
                                      uint8 values in the range [0, 255].
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the Hematoxylin and DAB 
                                       channels as float32 numpy arrays of shape (H, W).
    """
    # Convert uint8 to float32 in [0, 1] range
    image_float = image_rgb_uint8.astype(np.float32) / 255.0
    
    # Convert to OD space
    epsilon = 1e-6
    od = -np.log10(image_float + epsilon)
    
    # Ruifrok H-DAB stain matrix
    stain_matrix = np.array([
        [0.6500286, 0.7040310, 0.2860126],
        [0.2688606, 0.5700937, 0.7767574],
        [0.7110272, 0.4234194, 0.5615672]
    ], dtype=np.float32)
    
    # Inverse matrix
    stain_matrix_inv = np.linalg.inv(stain_matrix)
    
    # Deconvolve: OD (H, W, 3) @ M_inv (3, 3) -> Stains (H, W, 3)
    stains = np.dot(od, stain_matrix_inv)
    
    # Extract channels
    h_channel = stains[:, :, 0]
    dab_channel = stains[:, :, 1]
    
    # Clamp to zero
    h_channel = np.clip(h_channel, a_min=0.0, a_max=None)
    dab_channel = np.clip(dab_channel, a_min=0.0, a_max=None)
    
    return h_channel, dab_channel
