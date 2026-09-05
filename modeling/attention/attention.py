import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def build_2d_sincos_pos_embed_interleaved(H: int, W: int, C: int, *, device=None, dtype=torch.float32):
    """
    DETR-like
    Return [H*W, C] 2D sine-cos positional embedding with DETR-like interleaving:
      - First C/2 channels = Y axis (even idx -> sin, odd idx -> cos)
      - Last  C/2 channels = X axis (even idx -> sin, odd idx -> cos)
    Works for any C >= 2; if C is odd, last channel is dropped.
    """
    assert H > 0 and W > 0
    if C < 2:
        raise ValueError("C must be >= 2")

    device = torch.device(device) if device is not None else None
    # Use even number of features per axis; if C is odd, we'll trim later
    half = C // 2

    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H,W]
    yy = yy.reshape(-1, 1)  # [HW,1]
    xx = xx.reshape(-1, 1)  # [HW,1]

    # DETR-style frequency schedule
    dim_t = torch.arange(half, device=device, dtype=dtype)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / half)  # [half]

    # scale coords
    pos_y = yy / dim_t  # [HW, half]
    pos_x = xx / dim_t  # [HW, half]

    # interleave even->sin, odd->cos
    def interleave_sin_cos(pos):  # [HW, half] -> [HW, half]
        even = pos[:, 0::2].sin()
        odd  = pos[:, 1::2].cos()
        out = torch.empty(pos.size(0), even.size(1) + odd.size(1), device=pos.device, dtype=pos.dtype)
        out[:, 0::2] = even
        out[:, 1::2] = odd
        return out

    pe_y = interleave_sin_cos(pos_y)  # [HW, half]
    pe_x = interleave_sin_cos(pos_x)  # [HW, half]
    pe = torch.cat([pe_y, pe_x], dim=1)  # [HW, 2*half] == [HW, C or C-1]

    # If C is odd, drop the last column to match C exactly
    if pe.size(1) > C:
        pe = pe[:, :C]
    return pe.contiguous()  # [HW, C]


class ClassAttentionBlock(nn.Module):
    """
    Class-wise attention over a spatial feature map, returning:
      - class_features: [B, num_classes, C]
      - attn_maps:      [B, num_classes, H, W]  (CAM-like; min-max normalized if enabled)

    Inputs
    ------
    features: Tensor[B, C, H, W]

    Args
    ----
    feature_dim: channel dimension C of the input feature map
    num_classes: number of class queries
    class_embed_dim: dimension of class embeddings (defaults to feature_dim)
    num_heads: number of attention heads
    use_learnable_embeddings: if False with predefined_class_embeddings, embeddings are frozen
    predefined_class_embeddings: optional tensor [num_classes, class_embed_dim]
    use_pos_embed: add a fixed DETR-style 2D sin-cos positional embedding to the
        spatial keys/values before attention (default False, matches the original
        no-position-embedding behavior)
    normalize_attn_maps: min-max normalize attn_maps per (B, class) at the end of
        forward, on top of the softmax normalization (default False, matches the
        original behavior)
    """

    def __init__(self,
                 feature_dim: int,
                 num_classes: int,
                 class_embed_dim: Optional[int] = None,
                 num_heads: int = 4,
                 use_learnable_embeddings: bool = True,
                 predefined_class_embeddings: Optional[torch.Tensor] = None,
                 use_pos_embed: bool = False,
                 normalize_attn_maps: bool = False):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim or feature_dim
        self.num_heads = num_heads
        self.use_learnable_embeddings = use_learnable_embeddings
        self.use_pos_embed = use_pos_embed
        self.normalize_attn_maps = normalize_attn_maps

        # Class embeddings
        if predefined_class_embeddings is not None:
            assert predefined_class_embeddings.shape == (num_classes, self.class_embed_dim), \
                "Predefined class embeddings must have shape [num_classes, class_embed_dim]"
            self.class_embeddings = nn.Parameter(
                predefined_class_embeddings, requires_grad=use_learnable_embeddings
            )
        else:
            self.class_embeddings = nn.Parameter(
                torch.randn(num_classes, self.class_embed_dim),
                requires_grad=use_learnable_embeddings
            )

        # Project class embeddings to match feature dimension if needed
        self.class_proj = (nn.Linear(self.class_embed_dim, feature_dim) if self.class_embed_dim != feature_dim else nn.Identity())
        
        # Multihead attention (batch_first=True => [B, L, C])
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )

    @staticmethod
    def _cosine_maps(features: torch.Tensor, class_queries: torch.Tensor) -> torch.Tensor:
        """
        features:      [B, C, H, W]
        class_queries: [B, num_classes, C]
        returns:       [B, num_classes, H, W] in [-1, 1]
        """
        B, C, H, W = features.shape
        # L2-normalize features per spatial location
        f = F.normalize(features, dim=1)                     # [B, C, H, W]
        # L2-normalize class queries
        q = F.normalize(class_queries, dim=-1)               # [B, num_classes, C]
        # Broadcast dot-product over H,W
        cos = (f.unsqueeze(1) * q.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)  # [B, num_classes, H, W]
        return cos


    def forward(self,
                features: torch.Tensor,
                upsample_to: Optional[Tuple[int, int]] = None):
        """
        Args:
            features: Tensor[B, C, H, W]
            upsample_to: optionally upsample attention maps to (H_out, W_out) for visualization

        Returns:
            class_features: [B, num_classes, C]
            attn_maps:      [B, num_classes, H, W]
        """
        B, C, H, W = features.shape

        # [B, HW, C]
        features_seq = features.view(B, C, H * W).permute(0, 2, 1)

        # Optional fixed 2D positional embedding added to keys/values (no params)
        if self.use_pos_embed:
            pe = build_2d_sincos_pos_embed_interleaved(H, W, C, device=features_seq.device)  # [HW, C]
            kv = features_seq + pe.unsqueeze(0)  # [B, HW, C]
        else:
            kv = features_seq

        # Class queries: [B, num_classes, C]
        class_queries = self.class_proj(self.class_embeddings)  # [num_classes, C]
        class_queries = class_queries.unsqueeze(0).expand(B, -1, -1)

        # --- Attention ---
        # Try to get per-head weights (newer PyTorch); fall back to averaged (older PyTorch).
        try:
            attn_output, attn_weights = self.attention(
                query=class_queries,          # [B, num_classes, C]
                key=kv,                       # [B, HW, C]
                value=kv,                     # [B, HW, C]
                need_weights=True,
                average_attn_weights=False,   # <- keep heads
            )
        except TypeError:
            # Older signature: we already pass average_attn_weights=False via positional args
            attn_output, attn_weights = self.attention(class_queries, kv, kv, None, True, None, False)

        # Build CAM-like maps (avg over heads if per-head present)
        if attn_weights.dim() == 4:
            # mean of top-k heads (k=2) preserves multi-instance focus
            k = min(3, attn_weights.size(1))
            topk, _ = torch.topk(attn_weights, k=k, dim=1)   # [B,k,C,HW]
            attn_weights = topk.mean(dim=1)                  # [B,C,HW]
        elif attn_weights.dim() != 3:
            raise RuntimeError(f"Unexpected attn_weights shape: {attn_weights.shape}")

        # Re-normalize over HW
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Now attn_weights is [B, num_classes, HW]
        attn_maps = attn_weights.view(B, self.num_classes, H, W)

        # Optional min-max normalization on top of softmax, per (B, class) — nicer for visualization
        if self.normalize_attn_maps:
            attn_maps = (attn_maps - attn_maps.amin((-2, -1), keepdim=True)) \
                / (attn_maps.amax((-2, -1), keepdim=True) - attn_maps.amin((-2, -1), keepdim=True) + 1e-6)

        # --- Signed cosine maps (diagnostic, class selectivity) ---
        class_features = attn_output                    # [B, num_classes, C]
        cos_maps = self._cosine_maps(features, class_features)  # [B, num_classes, H, W]

        # Optional upsampling
        if upsample_to is not None:
            attn_maps = F.interpolate(attn_maps, size=upsample_to, mode='bilinear', align_corners=False)
            cos_maps  = F.interpolate(cos_maps,  size=upsample_to, mode='bilinear', align_corners=False)

        return class_features, attn_maps, cos_maps
