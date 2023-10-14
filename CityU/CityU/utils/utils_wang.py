import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import logging
from einops import rearrange
import torch.nn.functional as F


def cal_metrics(angRes_in, label, out, ):
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=angRes_in, a2=angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=angRes_in, a2=angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(),
                                                                out_y[b, u, :, v, :].numpy())
                SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True, )
                print(' Valid----scene, PSNR---%f, u---%f, v---%f' % (PSNR[b, u, v], u, v))
                print(' Valid----scene, SSIM---%f, u---%f, v---%f' % (SSIM[b, u, v], u, v))

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean

def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF


def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y
