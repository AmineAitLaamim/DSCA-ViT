import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Gated Token Fusion block for fusing two token sequences.
    
    Args:
        embed_dim (int): Embedding dimension. Default is 768.
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear layer for CLS token fusion
        self.cls_fusion = nn.Linear(embed_dim * 2, embed_dim)
        
        # Linear layer for calculating gate values for patch tokens
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, h_tokens: torch.Tensor, d_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses two token sequences H_final and D_final.
        
        Args:
            h_tokens (torch.Tensor): First token sequence of shape (B, N, embed_dim)
            d_tokens (torch.Tensor): Second token sequence of shape (B, N, embed_dim)
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - fused_tokens (torch.Tensor): Fused token sequence of shape (B, N, embed_dim)
                - gate_values (torch.Tensor): Gate values for patch tokens of shape (B, N-1, embed_dim)
        """
        # CLS token fusion
        cls_h = h_tokens[:, 0:1, :] # (B, 1, embed_dim)
        cls_d = d_tokens[:, 0:1, :] # (B, 1, embed_dim)
        cls_concat = torch.cat([cls_h, cls_d], dim=-1) # (B, 1, embed_dim * 2)
        fused_cls = self.cls_fusion(cls_concat) # (B, 1, embed_dim)
        
        # Patch token fusion
        patch_h = h_tokens[:, 1:, :] # (B, N-1, embed_dim)
        patch_d = d_tokens[:, 1:, :] # (B, N-1, embed_dim)
        patch_concat = torch.cat([patch_h, patch_d], dim=-1) # (B, N-1, embed_dim * 2)
        gate_values = torch.sigmoid(self.gate_proj(patch_concat)) # (B, N-1, embed_dim)
        fused_patches = gate_values * patch_h + (1 - gate_values) * patch_d # (B, N-1, embed_dim)
        
        fused_tokens = torch.cat([fused_cls, fused_patches], dim=1) # (B, N, embed_dim)
        return fused_tokens, gate_values


class RefinementBlock(nn.Module):
    """
    Refinement Block: A single standard transformer block (self-attention + FFN).
    
    Args:
        embed_dim (int): Embedding dimension. Default is 768.
        num_heads (int): Number of attention heads. Default is 12.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default is 4.0.
        dropout (float): Dropout rate. Default is 0.0.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initializes weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies self-attention and FFN.
        
        Args:
            x (torch.Tensor): Input tokens of shape (B, N, embed_dim)
            
        Returns:
            torch.Tensor: Refined tokens of shape (B, N, embed_dim)
        """
        # Pre-norm style
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x


class ClassificationHead(nn.Module):
    """
    Classification Head for DSCA-ViT.
    Extracts CLS token and global average pooling (GAP) of patch tokens, concatenates them,
    and applies a classifier to predict class logits.
    
    Args:
        num_classes (int): Number of classes for classification.
        embed_dim (int): Embedding dimension. Default is 768.
        dropout (float): Dropout rate in the classifier. Default is 0.1.
    """
    def __init__(self, num_classes: int, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extracts features and computes classification logits.
        
        Args:
            tokens (torch.Tensor): Refined tokens of shape (B, N, embed_dim)
            
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        cls_token = tokens[:, 0, :] # (B, embed_dim)
        gap = tokens[:, 1:, :].mean(dim=1) # (B, embed_dim)
        
        z = torch.cat([cls_token, gap], dim=-1) # (B, embed_dim * 2)
        z = self.norm(z)
        logits = self.classifier(z) # (B, num_classes)
        return logits
