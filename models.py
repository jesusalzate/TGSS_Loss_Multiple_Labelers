import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import ResidualUnit, Convolution
from monai.networks.layers import Act, Norm
from typing import Sequence,Union
from monai.networks.nets import UNet

class CustomUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        n_scorers: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int],int] = 3,
        up_kernel_size: Union[Sequence[int],int] = 3,
        num_res_units: int = 0,
        act: Union[tuple , str] = Act.PRELU,
        norm: Union[tuple , str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()
        
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[-1],  # output channels of the last layer in the UNet
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )
        
        self.final_conv_xy = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            act=act,
            norm=norm,
            bias=bias,
        )
        
        self.final_conv_lambda = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[-1],
            out_channels=n_scorers,
            kernel_size=1,
            act=act,
            norm=norm,
            bias=bias,
        )

    def forward(self, x):
        x = self.unet(x)
        xy = self.final_conv_xy(x)
        x_lambda = self.final_conv_lambda(x)
        return torch.cat((xy, x_lambda), dim=1)