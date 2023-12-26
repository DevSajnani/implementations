import torch
from torch import nn
import copy
import einops


class PatchEmbeddings(nn.Module):
  def __init__(self, patch_size, hidden_d, in_channels):
    super().__init__()
    self.patch_size = patch_size
    self.patcher = nn.Conv2d(in_channels=in_channels,
                   out_channels=hidden_d,
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)

  def forward(self, x):
    assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0, 'Height and width must be divisible by patch size' 
    x_patched = self.patcher(x)
    x_flattened = einops.rearrange(x_patched, 'b d n1 n2 -> b (n1 n2) d') # ([batch_sz, num_patches, D])
    return x_flattened
  

class LearnedPositionEmbeddings(nn.Module): 
  def __init__(self, b, d, n): 
    super().__init__()
    self.tokens = nn.Parameter(torch.rand(b, 1, d),
                           requires_grad=True) 
    self.positional_encoding = nn.Parameter(torch.rand(b, n+1, d),
                                  requires_grad=True)

  def forward(self, x): 
    x_token = torch.cat((self.tokens, x), dim=1)
    x_position = x_token + self.positional_encoding
    return x_position



class MultiHeadedSelfAttention(nn.Module): 
  def __init__(self, d, heads, dropout): 
    super().__init__()
    self.layer_norm = nn.LayerNorm(normalized_shape=d)
    self.multiheadedAtt = nn.MultiheadAttention(embed_dim=d, num_heads=heads, dropout=dropout, batch_first=True)

  def forward(self, x): 
    x = self.layer_norm(x)
    x_attn, _ = self.multiheadedAtt(query=x, 
                                    key=x, 
                                    value=x, 
                                    need_weights=False) 
    return x_attn
  


class MLPLayer(nn.Module): 
  def __init__(self, d, mlp_size, dropout): 
    super().__init__()
    self.layerNorm = nn.LayerNorm(normalized_shape=d)
    self.mlp = nn.Sequential(
        nn.Linear(in_features=d, out_features=mlp_size), 
        nn.GELU(), 
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size, out_features=d),
        nn.Dropout(p=dropout)
    )

  def forward(self, x): 
    x = self.layerNorm(x)
    x = self.mlp(x)
    return x
  

class TransformerBlock(nn.Module): 
  def __init__(self, d, mlp_size, heads, dropout): 
    super().__init__()
    self.msa = MultiHeadedSelfAttention(d, heads, dropout)
    self.mlp = MLPLayer(d, mlp_size, dropout)

  def forward(self, x): 
    x = self.msa(x) + x
    x = self.mlp(x) + x
    return x 
  
class ClassificationHead(nn.Module): 
  def __init__(self, d, n_classes): 
    super().__init__()
    self.classify = nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(in_features=d, out_features=n_classes)
    )

  def forward(self, x): 
    return self.classify(x)
  
class ViT(nn.Module): 
  def __init__(self, transformer_layer, patcher, embed, n_layers, classifier): 
    super().__init__()
    self.n = n_layers
    self.transformer = nn.ModuleList([copy.deepcopy(transformer_layer) for _ in range(n_layers)])
    self.patch_emb = patcher
    self.pos_emb = embed
    self.classify = classifier

  def forward(self, x): 
    x = self.patch_emb(x)
    x = self.pos_emb(x)
    for i in range(self.n):
        x = self.transformer[i](x)
    x = self.classify(x[:, 0])
    return x
  
