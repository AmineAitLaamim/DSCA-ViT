import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class SpatialBiasMatrix(nn.Module):
    """
    Spatially-Biased Matrix to add inductive bias based on the physical distance 
    between patches. The CLS token has no spatial bias.
    """
    def __init__(self, num_tokens: int = 197, gamma: float = 0.1, beta: float = 1.0):
        """
        Args:
            num_tokens: Total number of tokens (1 CLS + patch tokens).
            gamma: Hyperparameter for distance penalty.
            beta: Hyperparameter for self-attention bonus.

        Note on gamma:
            gamma controls the strength of the spatial locality prior.
            Large values (>0.3) effectively mask distant tokens because the bias
            is added before the softmax. A small value (0.1) provides a soft
            inductive bias while still allowing the model to learn non-local
            correspondences when useful.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.grid_size = int(math.sqrt(num_tokens - 1))
        
        if self.grid_size * self.grid_size != num_tokens - 1:
            raise ValueError("Number of patch tokens must be a perfect square.")
        
        bias = torch.zeros(num_tokens, num_tokens)
        
        for i in range(1, num_tokens):
            for j in range(1, num_tokens):
                if i == j:
                    bias[i, j] = beta
                else:
                    row_i, col_i = (i - 1) // self.grid_size, (i - 1) % self.grid_size
                    row_j, col_j = (j - 1) // self.grid_size, (j - 1) % self.grid_size
                    dist = math.sqrt((row_i - row_j) ** 2 + (col_i - col_j) ** 2)
                    bias[i, j] = -gamma * dist
                    
        # CLS token (index 0) naturally has 0 bias to and from other tokens 
        # since torch.zeros initializes to 0.
        
        self.bias_matrix = nn.Parameter(bias)

        # ------------------------------------------------------------
        # Print initialization values for verification
        # ------------------------------------------------------------
        max_dist = math.sqrt((self.grid_size - 1) ** 2 + (self.grid_size - 1) ** 2)
        max_penalty = -gamma * max_dist
        print("Spatial Bias Initialization")
        print("---------------------------")
        print(f"beta         : {beta}")
        print(f"gamma        : {gamma}")
        print(f"max distance : {max_dist:.2f}")
        print(f"max penalty  : {max_penalty:.2f}")
        print()
        
    def forward(self) -> torch.Tensor:
        """
        Returns:
            The learnable spatial bias matrix of shape (num_tokens, num_tokens).
        """
        return self.bias_matrix


class CrossAttentionLayer(nn.Module):
    """
    Single direction cross-attention layer applying queries from a source stream
    and keys/values from a context stream, with spatial bias.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        """
        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")
            
        self.norm_source = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.attn_weights: Optional[torch.Tensor] = None
        
    def forward(
        self, 
        source: torch.Tensor, 
        context: torch.Tensor, 
        spatial_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            source: Source tokens of shape (B, N, C).
            context: Context tokens of shape (B, M, C).
            spatial_bias: Optional spatial bias matrix of shape (N, M).
            
        Returns:
            Updated source tokens of shape (B, N, C).
        """
        batch_size, num_source_tokens, _ = source.shape
        _, num_context_tokens, _ = context.shape
        
        # Pre-norm
        normed_source = self.norm_source(source)
        normed_context = self.norm_context(context)
        
        # Projections and reshaping for multi-head attention
        q = self.q_proj(normed_source).reshape(
            batch_size, num_source_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        
        k = self.k_proj(normed_context).reshape(
            batch_size, num_context_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        
        v = self.v_proj(normed_context).reshape(
            batch_size, num_context_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        
        # Compute attention logits
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add spatial bias before softmax
        if spatial_bias is not None:
            # spatial_bias is (N, M)
            # attn_logits is (B, num_heads, N, M)
            attn_logits = attn_logits + spatial_bias.unsqueeze(0).unsqueeze(0)
            
        # Attention weights
        attn = attn_logits.softmax(dim=-1)
        self.attn_weights = attn.detach()
        attn = self.attn_drop(attn)
        
        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_source_tokens, self.embed_dim)
        
        # Output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Residual connection
        return source + x


class CrossAttentionFFN(nn.Module):
    """
    Standard Transformer Feed-Forward Network with pre-norm and residual connection.
    """
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 3072, dropout: float = 0.0):
        """
        Args:
            embed_dim: Input and output token dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens of shape (B, N, C).
            
        Returns:
            Output tokens of shape (B, N, C).
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + x


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention module enabling symmetric information exchange
    between two parallel streams (e.g., histology and DAPI), enriched with spatial bias.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_tokens: int = 197,
        ffn_hidden_dim: int = 3072,
        dropout: float = 0.0,
        gamma: float = 0.1,
        beta: float = 1.0
    ):
        """
        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
            num_tokens: Total number of tokens (for spatial bias).
            ffn_hidden_dim: Hidden dimension for the FFN layers.
            dropout: Dropout probability.
            gamma: Distance penalty for spatial bias.
                   Small values (e.g., 0.1) provide a soft locality prior;
                   large values (>0.3) effectively hard-mask distant tokens.
            beta: Self-attention bonus for spatial bias.
        """
        super().__init__()
        self.spatial_bias = SpatialBiasMatrix(
            num_tokens=num_tokens, gamma=gamma, beta=beta
        )
        
        self.cross_attn_h = CrossAttentionLayer(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attn_d = CrossAttentionLayer(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        
        self.ffn_h = CrossAttentionFFN(
            embed_dim=embed_dim, hidden_dim=ffn_hidden_dim, dropout=dropout
        )
        self.ffn_d = CrossAttentionFFN(
            embed_dim=embed_dim, hidden_dim=ffn_hidden_dim, dropout=dropout
        )
        
    def forward(
        self, h_tokens: torch.Tensor, d_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_tokens: Tokens from the histology stream, shape (B, N, C).
            d_tokens: Tokens from the DAPI stream, shape (B, N, C).
            
        Returns:
            Tuple of updated (h_tokens, d_tokens).
        """
        bias = self.spatial_bias()
        
        # Parallel cross-attention computation
        h_updated = self.cross_attn_h(source=h_tokens, context=d_tokens, spatial_bias=bias)
        d_updated = self.cross_attn_d(source=d_tokens, context=h_tokens, spatial_bias=bias)
        
        # Feed-forward networks
        h_out = self.ffn_h(h_updated)
        d_out = self.ffn_d(d_updated)
        
        return h_out, d_out
