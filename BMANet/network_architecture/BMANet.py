from einops import rearrange, reduce
from timm.models.layers.activations import *
from bmanet.network_architecture.neural_network import SegmentationNetwork
from bmanet.utilities.nd_softmax import softmax_helper
# from model import MODEL
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class ConvBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
            trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = stride
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(
            self, spatial_dims: int, in_channels: int, out_channels: int,
            dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


def get_conv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        stride: Union[Sequence[int], int] = 1,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = False,
        conv_only: bool = True,
        is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
        kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
        kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def relative_pos_dis(height=32, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (3, D, H, W)
    coords_flatten = torch.flatten(coords, 1)  # (3, DHW)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3,DHW,DHW)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    dis = (relative_coords[:, :, 0].float() / height) ** 2 + (relative_coords[:, :, 1].float() / weight) ** 2
    return dis


class TransformerBlock(nn.Module):
    def __init__(self, input_x: int, input_y: int, hidden_size: int, num_heads: int,
                 dropout_rate: float = 0.0, isup=False):
        if hidden_size % num_heads != 0:
            print('Hidden size is ', hidden_size)
            print('Num heads is ', num_heads)
            raise ValueError('hidden_size should be divisible by num_heads.')
        super().__init__()
        self.norm = nn.BatchNorm3d(hidden_size)  # nn.LayerNorm(hidden_size)
        self.att_block = Attention(hidden_size=hidden_size, num_heads=num_heads, attn_drop=dropout_rate, isup=isup,
                                   input_x=input_x, input_y=input_y)
        self.conv = get_conv_layer(3, hidden_size, hidden_size, kernel_size=3, stride=1, act='GELU', norm="BATCH")

    def forward(self, x):
        attn = self.att_block(self.norm(x))
        attn_skip = self.conv(attn)
        x = attn_skip
        return x


class Attention(nn.Module):

    def __init__(self, hidden_size, num_heads=4, attn_drop=0.1, isup=False, input_x=4, input_y=4):
        super().__init__()
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.isup = isup

        self.qkv = nn.Conv3d(hidden_size, hidden_size * 3, kernel_size=3, padding=1, bias=False)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=hidden_size*2)
        self.norm1 = nn.GroupNorm(num_groups=2, num_channels=hidden_size*2)
        self.reout = get_conv_layer(3, hidden_size, hidden_size, kernel_size=3, stride=1, act='RELU', norm="GROUP")
        if self.isup:
            self.dis = relative_pos_dis(input_x, input_y).to(device)
            self.headsita = nn.Parameter(torch.randn(num_heads), requires_grad=True)
            self.sig = nn.Sigmoid()

            self.to_out = nn.Sequential(
                nn.Conv3d(hidden_size, int(hidden_size // 2), kernel_size=1, padding=0, bias=False),
                nn.BatchNorm3d(int(hidden_size // 2)),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        temp = x
        qkv = self.qkv(x).chunk(3, dim=1)

        if self.isup:  #GDSA
            z = x.shape[2]
            q, k, v = map(lambda t: rearrange(t, 'b (g d) z h w -> b g z (h w) d', g=self.num_heads), qkv)
            k = k.transpose(-2, -1)
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            attn = (q @ k)

            factor = 1 / (2 * (self.sig(self.headsita) * (0.4 - 0.003) + 0.003) ** 2)
            dis = factor[:, None, None] * self.dis[None, :, :]
            dis = torch.exp(-dis)
            dis = dis / torch.sum(dis, dim=-1)[:, :, None]
            dis = torch.unsqueeze(dis, dim=1)
            dis = dis.repeat(1, z, 1, 1)
            attn = attn * dis[None, :, :, :, :]
            # attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            v = torch.nn.functional.normalize(v, dim=-1)
            x = (attn @ v)
            t = math.ceil(x.shape[3] ** (1 / 2))
            x = rearrange(x, 'b g z (h w) d -> b (g d) z h w', h=t, w=t)
        else:  # FGFF
            x = x.float()
            temp = x
            y = torch.fft.rfft2(x, dim=(3, 4))
            y_img = y.imag
            y_re = y.real
            y_img = torch.permute(y_img, (0, 2, 3, 4, 1))
            y_re = torch.permute(y_re, (0, 2, 3, 4, 1))

            y_img = self.linear1(y_img)
            y_re = self.linear2(y_re)
            y_img = torch.permute(y_img, (0, 4, 1, 2, 3))
            y_re = torch.permute(y_re, (0, 4, 1, 2, 3))
            q1, k1 = torch.chunk(y_img, 2, dim=1)
            q2, k2 = torch.chunk(y_re, 2, dim=1)
            q = torch.cat([q1, q2], dim=1)
            k = torch.cat([k1, k2], dim=1)

            q = self.norm(q)
            k = self.norm1(k)
            attn = (q * k)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_img, x_re = torch.chunk(attn, 2, dim=1)
            x = torch.complex(x_re, x_img)
            x = torch.fft.irfft2(x, dim=(3, 4)).float()
            x = self.reout(x)

        return x

class UpBlo(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
            trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.norm = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)
        self.act = get_act_layer(name='SELU')

    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = self.norm(out)
        out = self.act(out)
        return out


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class DWConv(nn.Module):
    def __init__(self, dim=768, out=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv3d(dim, out, 3, padding=1, groups=out)
        )

    def forward(self, x):
        x = self.dwconv(x)
        return x


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.05, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.dwconv = DWConv(hidden_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1)  # nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, D, H, W = x.shape
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)

        x = self.dwconv(x_s)
        x = self.act(x)
        x = self.drop(x)

        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)
        x = self.fc2(x_s)
        x = self.drop(x)
        return x


class BMANet(SegmentationNetwork):

    def __init__(self, dim_in=1, num_classes=1000, depths=[1, 2, 4, 2], stem_dim=24, embed_dims=[32, 48, 80, 168],
                 drop=0.):
        super().__init__()
        print('This is BMA-Net!')
        self.num_classes = num_classes
        self.final_nonlin = softmax_helper
        self.do_ds = False
        self.conv_op = nn.Conv3d
        assert num_classes > 0
        self.skip = nn.Sequential(
            get_conv_layer(3, dim_in, stem_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), dropout=drop,
                           conv_only=True),
            get_norm_layer(name='BATCH', spatial_dims=3, channels=stem_dim),
            get_act_layer(name='LEAKYRELU'))
        self.stage0 = nn.ModuleList([
            get_conv_layer(3, dim_in, embed_dims[0], kernel_size=(1, 2, 2), stride=(1, 2, 2), dropout=drop,
                           conv_only=True),
            get_norm_layer(name='BATCH', spatial_dims=3, channels=embed_dims[0]),
            get_act_layer(name='LEAKYRELU')
        ])
        self.skip_down = nn.Sequential(
            get_conv_layer(3, embed_dims[0], embed_dims[3], kernel_size=(4, 8, 8), stride=(4, 8, 8), dropout=drop,
                           conv_only=True),
            get_norm_layer(name='BATCH', spatial_dims=3, channels=embed_dims[3]),
            get_act_layer(name='LEAKYRELU'))
        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            for j in range(depths[i]):
                layers.append(
                    UnetResBlock(spatial_dims=3, in_channels=embed_dims[i], out_channels=embed_dims[i], kernel_size=3,
                                 stride=1, norm_name='BATCH', act_name='LEAKYRELU'))
            self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
        self.downlayers = nn.ModuleList()

        for i in range(len(depths) - 1):
            if i < 1:
                kernel_size = (1, 2, 2)
                stride = (1, 2, 2)
            else:
                kernel_size = (2, 2, 2)
                stride = (2, 2, 2)
            down_layer = nn.Sequential(
                get_conv_layer(3, embed_dims[i], embed_dims[i + 1], kernel_size=kernel_size, stride=stride,
                               dropout=drop,
                               conv_only=True),
                get_norm_layer(name='BATCH', spatial_dims=3, channels=embed_dims[i + 1]),
                get_act_layer(name='LEAKYRELU')
            )
            self.downlayers.append(down_layer)

        self.translayersdown = nn.ModuleList()
        self.translayersup = nn.ModuleList()
        self.transup = nn.ModuleList()
        self.translayersdown.append(
            TransformerBlock(input_x=64, input_y=64, hidden_size=embed_dims[0], num_heads=4,
                             dropout_rate=0.1, isup=False))


        for i in range(1, len(depths)):
            if i == 1:
                kernel_size = (1,2,2)
                stride = (1,2,2)
            else:
                kernel_size = (2, 2, 2)
                stride = (2, 2, 2)
            self.translayersdown.append(
                nn.Sequential(
                    get_conv_layer(3, embed_dims[i-1], embed_dims[i], kernel_size=kernel_size,
                                   stride=stride, dropout=drop, conv_only=True),
                    get_norm_layer(name='BATCH', spatial_dims=3, channels=embed_dims[i]),
                    get_act_layer(name='GELU'),
                    TransformerBlock(input_x=64, input_y=64,  hidden_size=embed_dims[i], num_heads=4,
                             dropout_rate=0.1,  isup=False)
                )
            )

        self.translayersup.append(
            TransformerBlock(input_x=8, input_y=8, hidden_size=embed_dims[3], num_heads=4, dropout_rate=0.1,
                              isup=True))

        for i in range(1, len(depths)):
            if i == len(depths) - 1:
                upsample_kernel_size = (1, 2, 2)
                self.translayersup.append(
                    UpBlo(spatial_dims=3, in_channels=embed_dims[len(depths) - i],
                          out_channels=embed_dims[len(depths) - i - 1],
                          kernel_size=3,
                          stride=1, upsample_kernel_size=upsample_kernel_size, norm_name='BATCH', act_name='GELU'))
            else:
                upsample_kernel_size = (2, 2, 2)
                self.translayersup.append(
                    UpBlo(spatial_dims=3, in_channels=embed_dims[len(depths)-i], out_channels=embed_dims[len(depths) - i - 1],
                          kernel_size=3,
                          stride=1, upsample_kernel_size=upsample_kernel_size, norm_name='BATCH', act_name='GELU'))

        self.fusion = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.fusion.append(nn.Sequential(
                shiftmlp(in_features=3 * embed_dims[i], out_features=embed_dims[i])
            ))

        self.up3 = UnetUpBlock(spatial_dims=3, in_channels=embed_dims[3], out_channels=embed_dims[2], kernel_size=3,
                               stride=2, upsample_kernel_size=2, norm_name='BATCH', act_name='LEAKYRELU')
        self.up2 = UnetUpBlock(spatial_dims=3, in_channels=embed_dims[2], out_channels=embed_dims[1], kernel_size=3,
                               stride=2, upsample_kernel_size=2, norm_name='BATCH', act_name='LEAKYRELU')
        self.up1 = UnetUpBlock(spatial_dims=3, in_channels=embed_dims[1], out_channels=embed_dims[0], kernel_size=3,
                               stride=(1, 2, 2), upsample_kernel_size=(1, 2, 2), norm_name='BATCH',
                               act_name='LEAKYRELU')
        self.up0 = UnetUpBlock(spatial_dims=3, in_channels=2 * embed_dims[0], out_channels=stem_dim, kernel_size=3,
                               stride=(1, 2, 2), upsample_kernel_size=(1, 2, 2), norm_name='BATCH',
                               act_name='LEAKYRELU')
        self.pro3 = ConvBlock(spatial_dims=3, in_channels=embed_dims[2], out_channels=num_classes, kernel_size=1,
                              stride=1, norm_name='BATCH', act_name='LEAKYRELU', dropout=0)
        self.pro2 = ConvBlock(spatial_dims=3, in_channels=embed_dims[1], out_channels=num_classes, kernel_size=1,
                              stride=1, norm_name='BATCH', act_name='LEAKYRELU', dropout=0)
        self.pro1 = ConvBlock(spatial_dims=3, in_channels=embed_dims[0], out_channels=num_classes, kernel_size=1,
                              stride=1, norm_name='BATCH', act_name='LEAKYRELU',
                              dropout=0)  # {'PRELU', 'CELU', 'TANH', 'LOGSOFTMAX', 'SWISH', 'ELU', 'RELU', 'MISH', 'MEMSWISH', 'RELU6', 'SELU', 'LEAKYRELU', 'SIGMOID', 'SOFTMAX', 'GELU'}.
        self.output = UnetOutBlock(spatial_dims=3, in_channels=stem_dim, out_channels=num_classes, dropout=0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm,
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'alpha', 'gamma', 'beta'}

    @torch.jit.ignore
    def no_ft_keywords(self):
        # return {'head.weight', 'head.bias'}
        return {}

    @torch.jit.ignore
    def ft_head_keywords(self):
        return {'head.weight', 'head.bias'}, self.num_classes

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()

    def check_bn(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
                m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)

    def forward(self, x):
        temp = x
        temp = self.skip(temp)
        for blk in self.stage0:
            x = blk(x)
        x0 = x
        skip_down = self.skip_down(x)
        y1down = self.translayersdown[0](x)
        for blk in self.stage1:
            x = blk(x)
        x1 = x
        x = self.downlayers[0](x)
        for blk in self.stage2:
            x = blk(x)
        x2 = x
        x = self.downlayers[1](x)
        for blk in self.stage3:
            x = blk(x)
        x3 = x
        x = self.downlayers[2](x)
        for blk in self.stage4:
            x = blk(x)
        x4 = x
        y4up = self.translayersup[0](skip_down)
        y3up = self.translayersup[1](y4up)
        y2up = self.translayersup[2](y3up)
        y1up = self.translayersup[3](y2up)

        y2down = self.translayersdown[1](y1down)
        y3down = self.translayersdown[2](y2down)
        y4down = self.translayersdown[3](y3down)

        x4 = torch.cat([x4, y4down, y4up], dim=1)
        x4 = self.fusion[3](x4)
        x3 = torch.cat([x3, y3down, y3up], dim=1)
        x3 = self.fusion[2](x3)
        x2 = torch.cat([x2, y2down, y2up], dim=1)
        x2 = self.fusion[1](x2)
        x1 = torch.cat([x1, y1down, y1up], dim=1)
        x1 = self.fusion[0](x1)

        out3 = self.up3(x4, x3)
        out2 = self.up2(out3, x2)
        out1 = self.up1(out2, x1)
        out10 = torch.cat([out1, x0], dim=1)
        out0 = self.up0(out10, temp)
        segout = []
        out = self.output(out0)
        segout.append(out)
        segout.append(self.pro1(out1))
        segout.append(self.pro2(out2))
        segout.append(self.pro3(out3))
        if self.do_ds:
            return segout
        else:
            return segout[0]


if __name__ == '__main__':

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    bs = 2
    reso = 224
    x = torch.randn(bs, 1, 32, 128, 128).to(device)
    fn = BMANet( dim_in=1, num_classes=3,
        depths=[2, 2, 8, 3], stem_dim=24, embed_dims=[24, 48, 96, 192], drop=0.1).to(device)

    fn.eval()
    y = fn(x)
    print(y.shape)
