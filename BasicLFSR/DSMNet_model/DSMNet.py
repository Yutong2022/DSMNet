import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import torch.nn.init as init
from .matching_module import *

class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


def make_layer(block, nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)


class AltFilter(nn.Module):
    def __init__(self, an):
        super(AltFilter, self).__init__()

        self.an = an
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.relu(self.spaconv(x))  # [N*an2,c,h,w]
        out = out.view(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * h * w, c, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.relu(self.angconv(out))  # [N*h*w,c,an,an]
        out = out.view(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * self.an * self.an, c, h, w)  # [N*an2,c,h,w]

        return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()


def conv(in_channels, out_channels, kernel_size=3, act=True, stride=1, groups=1, bias=True):
    m = []
    m.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=(kernel_size - 1) // 2, groups=groups, bias=bias))
    if act: m.append(nn.ReLU(inplace=True))
    return nn.Sequential(*m)


class PixelShuffle_Down(nn.Module):
    def __init__(self, scale=2):
        super(PixelShuffle_Down, self).__init__()
        self.scale = scale

    def forward(self, x):
        # assert h%scale==0 and w%scale==0
        b, c, h, w = x.size()
        x = x[:, :, :int(h - h % self.scale), :int(w - w % self.scale)]
        out_c = c * (self.scale ** 2)
        out_h = h // self.scale
        out_w = w // self.scale
        out = x.contiguous().view(b, c, out_h, self.scale, out_w, self.scale)
        return out.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_c, out_h, out_w)


def hard_knn(D, k):
    r"""
    input D: b m n
    output Idx: b m k
    """
    score, idx = torch.topk(D, k, dim=1, largest=False, sorted=True)

    return score, idx


class Encoder(nn.Module):
    def __init__(self, channels, an):
        super(Encoder, self).__init__()
        self.an = an
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=self.an, padding=self.an,
                                   bias=False)

    def forward(self, x):

        x = einops.rearrange(x, 'N (c u v) H W -> N c (H u) (W v)', u=self.an, v=self.an)
        buffer = self.init_conv(x)
        buffer = einops.rearrange(buffer, 'N c (H u) (W v) -> (N u v) c H W', u=self.an, v=self.an)

        return buffer


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        self.scale = args.scale_factor
        self.num_nbr = 5
        self.nf = args.nf
        self.an = args.angRes_in
        self.an2 = args.angRes_in*args.angRes_in

        self.encoder = Encoder(channels = self.nf, an = self.an)
        self.matching_group0 = matching_group(args)
        self.matching_group1 = matching_group(args)
        self.matching_group2 = matching_group(args)
        self.matching_group3 = matching_group(args)

        self.upsample = nn.Sequential(
            nn.Conv2d(self.nf, self.nf * self.scale ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(self.scale),
            nn.Conv2d(self.nf, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, lf_lr,  Lr_Info = None):
        # N, an2, H, W = lf_lr.size()
        x_upscale = F.interpolate(lf_lr, scale_factor=self.scale, mode='bilinear', align_corners=False)
        lf_lr = einops.rearrange(lf_lr, 'b 1 (u h) (v w) -> b (u v) h w', u=self.an, v=self.an)

        lf_fea = self.encoder(lf_lr)  # N*an2, 64, H, W
        lf_fea = self.matching_group0(lf_fea)
        lf_fea = self.matching_group1(lf_fea)
        lf_fea = self.matching_group2(lf_fea)
        lf_fea = self.matching_group3(lf_fea)

        # out
        lf_fea = einops.rearrange(lf_fea, '(N u v) c H W -> N c (u H) (v W)', u=self.an, v=self.an)

        out = self.upsample(lf_fea) + x_upscale
        # out = einops.rearrange(out, 'N c (u H) (v W) -> N (c u v) H W', u=self.an, v=self.an)

        return out


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        self.kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        self.kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.criterion_Loss = torch.nn.L1Loss()
        self.angRes = args.angRes
        self.loss_weight = args.angRes

    def forward(self, SR, HR):

        self.weight_h = nn.Parameter(data=self.kernel_h, requires_grad=False).to(SR.device)
        self.weight_v = nn.Parameter(data=self.kernel_v, requires_grad=False).to(SR.device)

        # yv
        SR_eyv = einops.rearrange(SR, 'b c (a1 h) (a2 w) -> (b h a1) c w a2', a1=self.angRes, a2=self.angRes)
        HR_eyv = einops.rearrange(HR, 'b c (a1 h) (a2 w) -> (b h a1) c w a2', a1=self.angRes, a2=self.angRes)
        SR_eyv_v = F.conv2d(SR_eyv, self.weight_v, padding=2)
        HR_eyv_v = F.conv2d(HR_eyv, self.weight_v, padding=2)
        l1 = self.criterion_Loss(SR_eyv_v, HR_eyv_v)
        SR_eyv_h = F.conv2d(SR_eyv, self.weight_h, padding=2)
        HR_eyv_h = F.conv2d(HR_eyv, self.weight_h, padding=2)
        l2 = self.criterion_Loss(SR_eyv_h, HR_eyv_h)

        # hu
        SR_ehu = einops.rearrange(SR, 'b c (a1 h) (a2 w) -> (b w a2) c h a1', a1=self.angRes, a2=self.angRes)
        HR_ehu = einops.rearrange(HR, 'b c (a1 h) (a2 w) -> (b w a2) c h a1', a1=self.angRes, a2=self.angRes)
        SR_ehu_v = F.conv2d(SR_ehu, self.weight_v, padding=2)
        HR_ehu_v = F.conv2d(HR_ehu, self.weight_v, padding=2)
        l3 = self.criterion_Loss(SR_ehu_v, HR_ehu_v)
        SR_ehu_h = F.conv2d(SR_ehu, self.weight_h, padding=2)
        HR_ehu_h = F.conv2d(HR_ehu, self.weight_h, padding=2)
        l4 = self.criterion_Loss(SR_ehu_h, HR_ehu_h)

        le = l1 + l2 + l3 + l4
        l0 = self.criterion_Loss(SR, HR)
        loss = 0.2*le + l0

        return loss

def weights_init(m):
    pass

