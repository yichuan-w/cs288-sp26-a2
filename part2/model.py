"""
Transformer model implementation from scratch.
Implements all components needed for a decoder-only transformer language model.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# Problem (linear): Implementing the linear module
# =============================================================================

class Linear(nn.Module):
    """
    Linear transformation layer: y = xW^T
    
    Note: We don't use bias in modern transformer implementations (like LLaMA).
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        Initialize linear layer.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Weight matrix of shape (d_out, d_in)
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply linear transformation: y = x @ W^T
        
        Args:
            x: Input tensor of shape (..., d_in)
        
        Returns:
            Output tensor of shape (..., d_out)
        """
        return x @ self.weight.t()


# =============================================================================
# Problem (embedding): Implement the embedding module
# =============================================================================

class Embedding(nn.Module):
    """
    Token embedding layer that maps token indices to dense vectors.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Embedding weight matrix of shape (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings from normal distribution."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Tensor of token indices of shape (batch, seq_len)
        
        Returns:
            Tensor of embeddings of shape (batch, seq_len, d_model)
        """
        return self.weight[token_ids]


# =============================================================================
# Problem (rmsnorm): Root Mean Square Layer Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simplification of LayerNorm that removes the mean centering
    and only normalizes by the root mean square of the activations.
    
    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension (size of last dimension)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMS normalization.
        
        RMSNorm(x) = x / RMS(x) * gamma
        where RMS(x) = sqrt(mean(x^2) + eps)
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# Problem (softmax): Implement softmax (used in attention)
# =============================================================================

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)

# =============================================================================
# SiLU activation (helper for SwiGLU)
# =============================================================================

def silu(x: Tensor) -> Tensor:
    """
    SiLU (Sigmoid Linear Unit) activation function.
    https://arxiv.org/abs/1702.03118
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor with SiLU applied element-wise
    """
    return x * torch.sigmoid(x)


# =============================================================================
# Problem (positionwise_feedforward): Implement the position-wise feed-forward network
# =============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    https://arxiv.org/pdf/2002.05202
    
    SwiGLU is a variant of the GLU (Gated Linear Unit) that uses SiLU activation.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize SwiGLU layer.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward layer
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gate projection: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff)
        # Down projection: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model)
        # Up projection: d_model -> d_ff
        self.w3 = Linear(d_model, d_ff)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of shape (..., d_model)
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Problem (rope): Implement RoPE (Rotary Position Embedding)
# =============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    RoPE encodes position information by rotating the query and key vectors
    in a way that makes the dot product depend on relative position.
    
    Mathematical Background:
    ------------------------
    RoPE applies a rotation matrix to pairs of dimensions. For a vector x at
    position m, each pair of dimensions (2i, 2i+1) is rotated by angle m * θ_i.
    
    The rotation angle for dimension pair i at position m is:
        angle_{m,i} = m * θ_i
        where θ_i = 1 / (base ^ (2i / d))
    
    For base=10000 and d=4 (head dimension):
        θ_0 = 1 / (10000 ^ (0/4)) = 1.0        (for dims 0,1)
        θ_1 = 1 / (10000 ^ (2/4)) = 0.01       (for dims 2,3)
    
    Implementation Details:
    -----------------------
    Instead of explicitly constructing rotation matrices, we use the identity:
    
        [cos(θ)  -sin(θ)] [x1]   [x1 * cos(θ) - x2 * sin(θ)]
        [sin(θ)   cos(θ)] [x2] = [x1 * sin(θ) + x2 * cos(θ)]
    
    This is equivalent to:
        x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
    
    Where rotate_half swaps and negates halves of the vector.
    
    Worked Example:
    ---------------
    For d_model=4, position m=2, base=10000:
    
    1. Compute inverse frequencies:
       inv_freq = [1/10000^(0/4), 1/10000^(2/4)] = [1.0, 0.01]
    
    2. Compute angles for position 2:
       angles = 2 * [1.0, 0.01] = [2.0, 0.02]
    
    3. Duplicate for paired dimensions:
       full_angles = [2.0, 0.02, 2.0, 0.02]  (via torch.cat([angles, angles]))
    
    4. Compute cos and sin:
       cos_cached[2] = [cos(2.0), cos(0.02), cos(2.0), cos(0.02)]
       sin_cached[2] = [sin(2.0), sin(0.02), sin(2.0), sin(0.02)]
    
    5. For input x = [x0, x1, x2, x3]:
       rotate_half(x) = [-x2, -x3, x0, x1]  (negate first half, swap)
       x_rotated = x * cos + rotate_half(x) * sin
    """
    
    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            d_model: Model dimension (head dimension for attention)
            max_seq_len: Maximum sequence length
            theta: Base for frequency computation (default: 10000.0)
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        # inv_freq shape: (d_model // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len)
    
    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values for positions up to seq_len."""
        # positions shape: (seq_len,)
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        
        # freqs shape: (seq_len, d_model // 2)
        freqs = torch.outer(positions, self.inv_freq)
        
        # Duplicate each frequency for the pair of dimensions
        # emb shape: (seq_len, d_model)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Rotate half the hidden dims of the input.
        
        This implements the "rotation" part of RoPE by rearranging dimensions.
        
        Operation:
            Split x into two halves along the last dimension:
                x1 = x[..., :d//2]  (first half)
                x2 = x[..., d//2:]  (second half)
            Return: [-x2, x1] concatenated
        
        Concrete Example:
            Input:  x = [a, b, c, d]  (d_model=4)
            Output: [-c, -d, a, b]
            
            Input:  x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  (d_model=6)
            x1 = [1.0, 2.0, 3.0]
            x2 = [4.0, 5.0, 6.0]
            Output: [-4.0, -5.0, -6.0, 1.0, 2.0, 3.0]
        
        Implementation:
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Apply rotary position embedding.
        
        Args:
            x: Input tensor of shape (batch, num_heads, seq_len, d_k)
               or (..., seq_len, d_model)
            token_positions: Position indices of shape (batch, seq_len) or (seq_len,)
        
        Returns:
            Tensor with rotary position embedding applied, same shape as input
        
        Formula:
            x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
        
        Implementation Steps:
            1. Index into precomputed cos/sin using token_positions:
               cos = self.cos_cached[token_positions]  # shape: (batch, seq_len, d_model)
               sin = self.sin_cached[token_positions]  # shape: (batch, seq_len, d_model)
            
            2. Handle broadcasting for 4D input (batch, heads, seq, d_k):
               If x has 4 dimensions, expand cos/sin with unsqueeze(1) to broadcast
               over the heads dimension.
            
            3. Apply the rotation formula:
               return x * cos + self._rotate_half(x) * sin
        
        Example:
            x = tensor of shape (2, 8, 10, 64)  # batch=2, heads=8, seq=10, d_k=64
            positions = tensor of shape (2, 10)  # positions for each sequence
            
            cos = self.cos_cached[positions]  # (2, 10, 64)
            cos = cos.unsqueeze(1)            # (2, 1, 10, 64) - broadcasts over heads
            
            x_rotated = x * cos + rotate_half(x) * sin  # (2, 8, 10, 64)
        """
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_model)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_model)
        
        # Handle 4D input (batch, heads, seq, d_k)
        if x.dim() == 4:
            cos = cos.unsqueeze(1)  # (batch, 1, seq_len, d_model)
            sin = sin.unsqueeze(1)  # (batch, 1, seq_len, d_model)
        
        return x * cos + self._rotate_half(x) * sin


def apply_rope(x: Tensor, d_model: int, theta: float, max_seq_len: int, token_positions: Tensor) -> Tensor:
    """
    Functional interface for applying RoPE.
    
    Args:
        x: Input tensor of shape (..., seq_len, d_model)
        d_model: Dimension of the model/head
        theta: RoPE base frequency
        max_seq_len: Maximum sequence length
        token_positions: Position indices
    
    Returns:
        Tensor with RoPE applied
    """
    rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
    rope = rope.to(x.device)
    return rope(x, token_positions)


# =============================================================================
# Problem (scaled_dot_product_attention): Implement scaled dot-product attention
# =============================================================================

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k)
              True values indicate positions to attend to, False positions are masked
    
    Returns:
        Attention output of shape (..., seq_len_q, d_v)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax
    attn_weights = softmax(scores, dim=-1)
    
    # Handle fully masked rows (all -inf -> NaN after softmax)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    # Compute output: attn_weights @ V
    return torch.matmul(attn_weights, V)


# =============================================================================
# Problem (multihead_self_attention): Implement causal multi-head self-attention
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer with causal masking.
    
    This implements the attention mechanism used in decoder-only transformers
    like GPT and LLaMA. It projects the input into queries, keys, and values,
    applies scaled dot-product attention with causal masking, and projects back.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Projection layers
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal (lower triangular) attention mask."""
        # mask[i, j] = True if j <= i (can attend to position j from position i)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len, x.device)
        
        # Apply attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask)  # (batch, num_heads, seq_len, d_k)
        
        # Reshape back to (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.output_proj(attn_out)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embedding (RoPE).
    
    This extends the basic multi-head attention by applying RoPE to the
    query and key vectors before computing attention scores.
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0):
        """
        Initialize multi-head self-attention with RoPE.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE base frequency
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Projection layers
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        # RoPE for query/key rotation
        self.rope = RotaryPositionEmbedding(self.d_k, max_seq_len, theta)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal (lower triangular) attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask
    
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply multi-head self-attention with RoPE.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices of shape (batch, seq_len)
                           If None, uses sequential positions [0, 1, 2, ...]
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Default to sequential positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE to Q and K
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len, x.device)
        
        # Apply attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.output_proj(attn_out)


# =============================================================================
# Problem (transformer_block): Implement the Transformer block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.
    
    Structure (Pre-LN / LLaMA-style):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            theta: RoPE base frequency
            eps: Epsilon for layer normalization
        """
        super().__init__()
        
        # Layer norms (Pre-LN)
        self.ln1 = RMSNorm(d_model, eps)
        self.ln2 = RMSNorm(d_model, eps)
        
        # Self-attention with RoPE
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply Transformer block (Pre-LN style).

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-LN style: x = x + Attention(RMSNorm(x))
        x = x + self.attn(self.ln1(x), token_positions)
        # x = x + FFN(RMSNorm(x))
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# Problem (transformer_lm): Implementing the Transformer LM
# =============================================================================

class TransformerLM(nn.Module):
    """
    Transformer Language Model (decoder-only, like GPT/LLaMA).
    
    Architecture:
        1. Token embedding
        2. N x Transformer blocks
        3. Final layer norm
        4. Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        """
        Initialize Transformer LM.
        
        Args:
            vocab_size: Size of vocabulary
            context_length: Maximum sequence/context length
            d_model: Model dimension
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            rope_theta: RoPE base frequency
            eps: Epsilon for layer normalization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_ln = RMSNorm(d_model, eps)
        
        # Output projection (to vocab size)
        self.output = Linear(d_model, vocab_size)
    
    def forward(self, token_ids: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Transformer LM.
        
        Args:
            token_ids: Input token indices of shape (batch, seq_len)
            token_positions: Optional position indices of shape (batch, seq_len)
                           If None, uses sequential positions [0, 1, 2, ...]
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Default to sequential positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 1. Token embeddings
        x = self.token_embeddings(token_ids)
        
        # 2. Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # 3. Final layer norm
        x = self.final_ln(x)
        
        # 4. Output projection to vocabulary
        logits = self.output(x)
        
        return logits
    
    def load_weights(self, state_dict: dict):
        """
        Load weights from a state dict.
        
        Args:
            state_dict: Dictionary mapping weight names to tensors
        """
        # Token embeddings
        if "token_embeddings.weight" in state_dict:
            self.token_embeddings.weight.data.copy_(state_dict["token_embeddings.weight"])
        
        # Output projection
        if "output.weight" in state_dict:
            self.output.weight.data.copy_(state_dict["output.weight"])
        
        # Final layer norm
        if "final_ln.weight" in state_dict:
            self.final_ln.weight.data.copy_(state_dict["final_ln.weight"])
        
        # Layer weights
        for layer_idx, layer in enumerate(self.layers):
            prefix = f"layers.{layer_idx}"
            
            # Layer norms
            if f"{prefix}.ln1.weight" in state_dict:
                layer.ln1.weight.data.copy_(state_dict[f"{prefix}.ln1.weight"])
            if f"{prefix}.ln2.weight" in state_dict:
                layer.ln2.weight.data.copy_(state_dict[f"{prefix}.ln2.weight"])
            
            # Attention projections
            if f"{prefix}.attn.q_proj.weight" in state_dict:
                layer.attn.q_proj.weight.data.copy_(state_dict[f"{prefix}.attn.q_proj.weight"])
            if f"{prefix}.attn.k_proj.weight" in state_dict:
                layer.attn.k_proj.weight.data.copy_(state_dict[f"{prefix}.attn.k_proj.weight"])
            if f"{prefix}.attn.v_proj.weight" in state_dict:
                layer.attn.v_proj.weight.data.copy_(state_dict[f"{prefix}.attn.v_proj.weight"])
            if f"{prefix}.attn.output_proj.weight" in state_dict:
                layer.attn.output_proj.weight.data.copy_(state_dict[f"{prefix}.attn.output_proj.weight"])
            
            # FFN weights
            if f"{prefix}.ffn.w1.weight" in state_dict:
                layer.ffn.w1.weight.data.copy_(state_dict[f"{prefix}.ffn.w1.weight"])
            if f"{prefix}.ffn.w2.weight" in state_dict:
                layer.ffn.w2.weight.data.copy_(state_dict[f"{prefix}.ffn.w2.weight"])
            if f"{prefix}.ffn.w3.weight" in state_dict:
                layer.ffn.w3.weight.data.copy_(state_dict[f"{prefix}.ffn.w3.weight"])


# =============================================================================
# Problem (transformer_accounting): Transformer LM resource accounting
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_flops_per_token(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> int:
    """
    Estimate the number of FLOPs per token for a forward pass.
    
    This is an approximation that counts multiply-accumulate operations (MACs).
    Each MAC is typically counted as 2 FLOPs.
    
    Args:
        vocab_size: Size of vocabulary
        context_length: Maximum sequence length (used for attention)
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
    
    Returns:
        Approximate FLOPs per token
    """
    flops = 0
    
    # Per layer:
    per_layer = 0
    
    # Attention: Q, K, V projections: 3 * (2 * d_model * d_model)
    per_layer += 3 * 2 * d_model * d_model
    
    # Attention score: Q @ K^T: 2 * d_model * context_length
    per_layer += 2 * d_model * context_length
    
    # Attention output: attn_weights @ V: 2 * d_model * context_length
    per_layer += 2 * d_model * context_length
    
    # Output projection: 2 * d_model * d_model
    per_layer += 2 * d_model * d_model
    
    # FFN (SwiGLU): w1: 2*d_model*d_ff, w3: 2*d_model*d_ff, w2: 2*d_ff*d_model
    per_layer += 2 * 2 * d_model * d_ff  # w1 and w3
    per_layer += 2 * d_ff * d_model      # w2
    
    flops += num_layers * per_layer
    
    # Output projection to vocab: 2 * d_model * vocab_size
    flops += 2 * d_model * vocab_size
    
    return flops


def estimate_memory_bytes(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    dtype_bytes: int = 4,  # float32 = 4 bytes
) -> int:
    """
    Estimate the memory required to store model parameters.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        d_ff: Feed-forward hidden dimension
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
    
    Returns:
        Approximate memory in bytes
    """
    num_params = 0
    num_params += vocab_size * d_model
    num_params += vocab_size * d_model
    num_params += d_model
    per_layer = (
        2 * d_model
        + 4 * d_model * d_model
        + 3 * d_model * d_ff
    )
    num_params += num_layers * per_layer
    return num_params * dtype_bytes
