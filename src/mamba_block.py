"""
Mamba Block implementation for the MARU project.

This module implements a pure PyTorch version of the Mamba architecture,
focusing on readability and educational value. It implements the Selective
Scan (S6) mechanism that makes Mamba's state-space model data-dependent.

Based on the architecture described in "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces" (Gu & Dao, 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import math


class MambaConfig:
    """Configuration class for Mamba block parameters."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True  # Use parallel scan vs sequential
    ):
        """
        Initialize Mamba configuration.
        
        Args:
            d_model: Model dimension (input/output dimension)
            d_state: State dimension (N in the paper)
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            dt_rank: Rank for dt projection (if None, uses d_model // 16)
            dt_min: Minimum value for dt initialization
            dt_max: Maximum value for dt initialization
            dt_init: Initialization method for dt ("random" or "constant")
            dt_scale: Scale factor for dt initialization
            bias: Whether to use bias in linear layers
            conv_bias: Whether to use bias in convolution
            pscan: Whether to use parallel scan (True) or sequential (False)
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank if dt_rank is not None else max(1, self.d_model // 16)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.bias = bias
        self.conv_bias = conv_bias
        self.pscan = pscan


def selective_scan_sequential(u, delta, A, B, C, D=None):
    """
    Sequential implementation of the selective scan operation.
    
    This is the "RNN-like" formulation that processes one timestep at a time.
    Used for educational purposes and when parallel scan is not available.
    
    Args:
        u: Input sequence (B, L, D)
        delta: Discretization step sizes (B, L, D)
        A: State transition matrix (D, N)
        B: Input matrix (B, L, N)
        C: Output matrix (B, L, N)
        D: Feedthrough matrix (D,) - optional
        
    Returns:
        y: Output sequence (B, L, D)
    """
    batch_size, seq_len, d_inner = u.shape
    _, _, d_state = B.shape
    
    # Initialize state
    h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
    
    # Discretize A using delta
    # A_discrete = exp(delta * A)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
    
    # Process sequence step by step
    outputs = []
    for t in range(seq_len):
        # Update state: h_t = A_discrete * h_{t-1} + delta * B * u
        h = deltaA[:, t] * h + delta[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
        
        # Compute output: y_t = C * h_t
        y_t = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1)  # (B, D)
        outputs.append(y_t)
    
    y = torch.stack(outputs, dim=1)  # (B, L, D)
    
    # Add feedthrough connection if provided
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)
    
    return y


class MambaBlock(nn.Module):
    """
    A single Mamba block implementing the Selective State Space Model.
    
    This block processes input sequences using a data-dependent state space model
    that can selectively focus on or ignore information based on input content.
    """
    
    def __init__(self, config: MambaConfig):
        """
        Initialize the Mamba block.
        
        Args:
            config: MambaConfig object specifying the block parameters
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_inner
        self.dt_rank = config.dt_rank
        
        # Input projection - projects d_model to 2 * d_inner for x and z paths
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)
        
        # Convolution layer for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=self.d_inner,  # Depthwise convolution
            padding=config.d_conv - 1
        )
        
        # Activation function
        self.activation = nn.SiLU()
        
        # SSM parameters
        # x_proj projects input to delta, B, C
        self.x_proj = nn.Linear(
            self.d_inner, 
            self.dt_rank + self.d_state * 2,  # dt_rank + d_state + d_state
            bias=False
        )
        
        # dt_proj projects dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize A parameter (state transition matrix)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        A_log = torch.log(A)  # Keep in log space for stability
        self.A_log = nn.Parameter(A_log)
        
        # Initialize D parameter (feedthrough/skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters according to Mamba paper."""
        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        ).clamp(min=self.config.dt_min)
        
        # Inverse of softplus
        dt = dt + torch.log(-torch.expm1(-dt))
        
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt)
        
        # Initialize other parameters with small random values
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mamba block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Input projection: split into x and z paths
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Apply convolution to x_path for local dependencies
        # Conv1d expects (B, C, L), so we need to transpose
        x_conv = self.conv1d(x_path.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = self.activation(x_conv)
        
        # Project to get SSM parameters
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2 * d_state)
        
        # Split into delta, B, C
        dt, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Process delta
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Apply selective scan
        y = selective_scan_sequential(x_conv, dt, A, B, C, self.D)
        
        # Apply gating with z
        y = y * self.activation(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


def create_mamba_block(d_model: int, **kwargs) -> MambaBlock:
    """
    Factory function to create a MambaBlock.
    
    Args:
        d_model: Model dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        MambaBlock instance
    """
    config = MambaConfig(d_model=d_model, **kwargs)
    return MambaBlock(config)


if __name__ == "__main__":
    # Simple test
    d_model = 64
    batch_size = 2
    seq_len = 32
    
    config = MambaConfig(d_model=d_model, d_state=16, expand=2)
    mamba = MambaBlock(config)
    
    x = torch.randn(batch_size, seq_len, d_model)
    y = mamba(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Config: d_model={config.d_model}, d_state={config.d_state}, d_inner={config.d_inner}")
    print("MambaBlock test passed!")
