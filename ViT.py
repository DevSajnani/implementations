import torch
from torch import nn
import einops

class Config: 
  class Config:
    d_model: int = 768
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    patch_size: int = 16
    in_channels: int = 3
    batch_size: int = 32
    height: int = 224 
    width: int = 224
    droput_msa: float = 0.0 
    dropout_mlp: float = 0.1
    num_classes: int = 3


class PatchEmbeddings(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.patch_size = self.cfg.patch_size
    hidden_d = self.cfg.d_model
    self.patcher = nn.Conv2d(in_channels=self.cfg.in_channels,
                   out_channels=hidden_d,
                   kernel_size=self.patch_size,
                   stride=self.patch_size,
                   padding=0)

  def forward(self, x):
    assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0, 'Height and width must be divisible by patch size' 
    x_patched = self.patcher(x)
    x_flattened = einops.rearrange(x_patched, 'b d n1 n2 -> b (n1 n2) d') # ([batch_sz, num_patches, D])
    return x_flattened
  

class LearnedPositionEmbeddings(nn.Module): 
  def __init__(self, cfg: Config): 
    super().__init__()
    self.cfg = cfg
    self.tokens = nn.Parameter(torch.rand(self.cfg.batch_size, 1, self.cfg.d_model),
                           requires_grad=True) 
    self.positional_encoding = nn.Parameter(torch.rand(self.cfg.batch_size, ((self.cfg.height * self.cfg.width) // (self.cfg.patch_size**2)), self.cfg.d_model),
                                  requires_grad=True)

  def forward(self, x): 
    x_token = torch.cat((self.tokens, x), dim=1)
    x_position = x_token + self.positional_encoding
    return x_position



class MultiHeadedSelfAttention(nn.Module): 
  def __init__(self, cfg: Config): 
    super().__init__()
    self.cfg = cfg
    self.layer_norm = nn.LayerNorm(normalized_shape=self.cfg.d_model)
    self.multiheadedAtt = nn.MultiheadAttention(embed_dim=self.cfg.d_model, num_heads=self.cfg.n_heads, dropout=self.cfg.droput_msa, batch_first=True)

  def forward(self, x): 
    x = self.layer_norm(x)
    x_attn, _ = self.multiheadedAtt(query=x, 
                                    key=x, 
                                    value=x, 
                                    need_weights=False) 
    return x_attn
  


class MLPLayer(nn.Module): 
  def __init__(self, cfg: Config): 
    super().__init__()
    self.cfg = cfg
    self.layerNorm = nn.LayerNorm(normalized_shape=self.cfg.d_model)
    self.mlp = nn.Sequential(
        nn.Linear(in_features=self.cfg.d_model, out_features=self.cfg.d_mlp), 
        nn.GELU(), 
        nn.Dropout(p=self.cfg.dropout_mlp),
        nn.Linear(in_features=self.cfg.d_mlp, out_features=self.cfg.d_model),
        nn.Dropout(p=self.cfg.dropout_mlp)
    )

  def forward(self, x): 
    x = self.layerNorm(x)
    x = self.mlp(x)
    return x
  

class TransformerBlock(nn.Module): 
  def __init__(self, cfg: Config): 
    super().__init__()
    self.cfg = cfg
    self.msa = MultiHeadedSelfAttention(cfg)
    self.mlp = MLPLayer(cfg)

  def forward(self, x): 
    x = self.msa(x) + x
    x = self.mlp(x) + x
    return x 
  
class ClassificationHead(nn.Module): 
  def __init__(self, cfg: Config): 
    super().__init__()
    self.cfg = cfg
    self.classify = nn.Sequential(
        nn.LayerNorm(self.cfg.d_model),
        nn.Linear(in_features=self.cfg.d_model, out_features=self.cfg.num_classes)
    )

  def forward(self, x): 
    return self.classify(x)
  
class ViT(nn.Module): 
  def __init__(self, cfg): 
    super().__init__()
    self.cfg = cfg
    self.transformer = nn.ModuleList([TransformerBlock(cfg) for _ in range(self.cfg.n_layers)])
    self.patch_emb = PatchEmbeddings(cfg)
    self.pos_emb = LearnedPositionEmbeddings(cfg)
    self.classify = ClassificationHead(cfg)

  def forward(self, x): 
    x = self.patch_emb(x)
    x = self.pos_emb(x)
    for i in range(self.n):
        x = self.transformer[i](x)
    x = self.classify(x[:, 0])
    return x
  
