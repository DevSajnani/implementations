import torch as t
from torch import nn
from torch import Tensor
import einops 
from typing import Tuple, List, Optional, Dict, Float, Int
from jaxtyping import Float, Int

class Config(): 
  latent_dim_size: int = 100
  img_size: int = 64 
  img_channels: int = 3
  hidden_channels: List[int] = [128, 256, 512]

class Generator(nn.Module):
    def __init__(
        self, cfg: Config
    ):
        
        n_layers = len(cfg.hidden_channels)
        assert cfg.img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        self.latent_dim_size = cfg.latent_dim_size
        self.img_size = cfg.img_size
        self.img_channels = cfg.img_channels
        self.hidden_channels = cfg.hidden_channels[::-1]

        first_height = self.img_size // (2 ** n_layers)
        first_size = self.hidden_channels[0] * (first_height ** 2)
        self.project_and_reshape = nn.Sequential(
            nn.Linear(self.latent_dim_size, first_size, bias=False),
            einops.rearrange("b (ic h w) -> b ic h w", h=first_height, w=first_height),
            nn.BatchNorm2d(self.hidden_channels[0]),
            nn.ReLU(),
        )

        in_channels = self.hidden_channels
        out_channels = self.hidden_channels[1:] + [self.img_channels]

        conv_layer_list = []
        for i, (c_in, c_out) in enumerate(zip(in_channels, out_channels)):
            conv_layer = [
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1),
                nn.ReLU() if i < n_layers - 1 else nn.Tanh()
            ]
            if i < n_layers - 1:
                conv_layer.insert(1, nn.BatchNorm2d(c_out))
            conv_layer_list.append(nn.Sequential(*conv_layer))

        self.hidden_layers = nn.Sequential(*conv_layer_list)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self, cfg: Config 
        ):
        n_layers = len(cfg.hidden_channels)
        assert cfg.img_size % (2 ** n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        self.img_size = cfg.img_size
        self.img_channels = cfg.img_channels
        self.hidden_channels = cfg.hidden_channels

        in_channels = [self.img_channels] + self.hidden_channels[:-1]
        out_channels = self.hidden_channels[1:]

        conv_layer_list = []
        negative_slope = 0.2
        for i, (c_in, c_out) in enumerate(zip(in_channels, out_channels)):
            conv_layer = [
                nn.Conv2d(c_in, c_out, 4, 2, 1),
                nn.LeakyReLU(negative_slope)
            ]
            if i > 0:
                conv_layer.insert(1, nn.BatchNorm2d(c_out))
            conv_layer_list.append(nn.Sequential(*conv_layer))

        self.hidden_layers = nn.Sequential(*conv_layer_list)

        final_height = self.img_size // (2 ** n_layers)
        final_size = self.hidden_channels[-1] * (final_height ** 2)
        self.classifier = nn.Sequential(
            einops.rearrange("b c h w -> b (c h w)"),
            nn.Linear(final_size, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x.squeeze() 


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self, cfg: Config
    ):
        super().__init__()
        self.latent_dim_size = cfg.latent_dim_size
        self.img_size = cfg.img_size
        self.img_channels = cfg.img_channels
        self.hidden_channels = cfg.hidden_channels
        self.netD = Discriminator(cfg = Config())
        self.netG = Generator(cfg = Config())
        initialize_weights(self)

def initialize_weights(model: nn.Module) -> None:
    for (name, module) in model.named_modules():
        if any([
            isinstance(module, Module) for Module in [nn.ConvTranspose2d, nn.Conv2d, nn.Linear]
        ]):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)