import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

compile_backend = "inductor"
g_device = "cuda"

# 1. Define the "Block Mask" logic (e.g., Causal Masking)
# FlexAttention uses a function to determine which tokens can attend to which.
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

class FlexAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Standard QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False, device=g_device)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=g_device)

    def forward(self, x):
        B, SeqLen, D = x.shape
        
        # 2. Project and reshape to (B, H, SeqLen, HeadDim)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to (B, Heads, SeqLen, HeadDim) for FlexAttention
        q = q.view(B, SeqLen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, SeqLen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, SeqLen, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Create the Block Mask (Only needs to be created once per shape/device)
        # This tells the kernel to skip calculating scores for masked tokens.
        block_mask = create_block_mask(
            causal_mask, 
            B, 
            self.num_heads, 
            SeqLen, 
            SeqLen, 
            device=x.device
        )

        # 4. Call flex_attention
        # score_mod=None implies standard softmax attention
        out = flex_attention(q, k, v, block_mask=block_mask)
        
        # 5. Reshape back
        out = out.transpose(1, 2).contiguous().view(B, SeqLen, D)
        return self.out_proj(out)


class ToyModule(nn.Module):
  def __init__(self):
      super().__init__()
      self.attn_layer = FlexAttentionLayer(d_model=1024, num_heads=64).to(device=g_device)
      self.forward = torch.compile(self.forward, backend=compile_backend)

  def forward(self, x):
      x2 = torch.sigmoid(x)
      v = self.attn_layer(x2)
      return torch.sigmoid(v)


model = ToyModule()
model.eval()

# Normal case:
x = torch.randn(8, 512, 1024, device=g_device).detach()

# Error case:
# Create dummy input (Batch=1024, SeqLen=1024, Dim=1024)
# x = torch.randn(1024, 1024, 1024, device=g_device).detach()

# Forward pass
with torch.no_grad():
    output = model(x)
print(f"\n==>Output shape: {output.shape}\n")
