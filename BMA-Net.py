import torch
import torch.nn as nn

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
        self.norm = nn.BatchNorm3d(hidden_size)
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
            self.dis = relative_pos_dis(input_x, input_y, sita=0.9).to(device)
            self.headsita = nn.Parameter(torch.randn(num_heads), requires_grad=True)
            self.sig = nn.Sigmoid()

            self.to_out = nn.Sequential(
                nn.Conv3d(hidden_size, int(hidden_size // 2), kernel_size=1, padding=0, bias=False),
                nn.BatchNorm3d(int(hidden_size // 2)),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        qkv = self.qkv(x).chunk(3, dim=1)

        if self.isup:  # GDSA
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
            attn = self.attn_drop(attn)
            v = torch.nn.functional.normalize(v, dim=-1)
            x = (attn @ v)
            t = math.ceil(x.shape[3] ** (1 / 2))
            x = rearrange(x, 'b g z (h w) d -> b (g d) z h w', h=t, w=t)
        else:  # FGFF
            x = x.float()
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


class DWConv(nn.Module):
    def __init__(self, dim=768, out=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv3d(dim, out, 3, padding=1, groups=out)
        )

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MSF(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.05, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.dwconv = DWConv(hidden_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1)
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














