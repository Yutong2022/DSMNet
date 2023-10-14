import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
from os.path import join
import math
import copy
import h5py
from skimage.metrics import structural_similarity as compare_ssim
from DSMNet_model.DSMNet import get_model
import utils.utils as utility
from PIL import Image
import warnings

import einops
from utils.utils_wang import *

warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------#
# Test settings
parser = argparse.ArgumentParser(description="PyTorch LFSSR-SAS testing")
parser.add_argument("--model_dir", type=str, default="", help="model dir")
parser.add_argument("--model_name", type=str, default="model_DSMNet_x2", help="model name")
parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--train_dataset", type=str, default="all", help="dataset for training")
parser.add_argument("--test_dataset", type=str, default="Kalantari", help="dataset for test")
parser.add_argument("--angular_num", type=int, default=7, help="Size of one angular dim")
parser.add_argument("--save_img", type=int, default=1, help="save image or not")
parser.add_argument("--mode", type=str, default="", help="SR factor")
parser.add_argument('--test_patch', action=utility.StoreAsArray, type=int, nargs='+', help="number of patches during testing")
parser.add_argument('--model', type=str, default='', help="SR model")
parser.add_argument('--nf', default=48, type=int, help='')
parser.add_argument("--angRes", type=int, default=7, help="Size of one angular dim")
parser.add_argument("--angRes_in", type=int, default=7, help="Size of one angular dim")
parser.add_argument("--k_nbr", type=int, default=6, help="Size of one angular dim")
parser.add_argument('--dataset_path', type=str, default='', help="SR model")


opt = parser.parse_args()
print(opt)
# -----------------------------------------------------------------------------------#
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, scale):
        super(DatasetFromHdf5, self).__init__()

        hf = h5py.File(file_path)
        self.GT_y = hf["/GT_y"]  # [N,aw,ah,h,w]
        self.LR_ycbcr = hf["/LR_ycbcr"]  # [N,ah,aw,3,h/s,w/s]

        self.scale = scale

    def __getitem__(self, index):
        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]

        gt_y = self.GT_y[index]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)

        lr_ycbcr = self.LR_ycbcr[index]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32) / 255.0)

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, h // self.scale, w // self.scale)

        lr_ycbcr_up = lr_ycbcr.view(1, -1, h // self.scale, w // self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bilinear',
                                                      align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, h, w)

        return gt_y, lr_ycbcr_up, lr_y

    def __len__(self):      
        return self.GT_y.shape[0]

# -----------------------------------------------------------------------------------#
def main():

    model_dir = opt.model_dir

    if not os.path.exists(model_dir):
        print('model folder is not found ')

    root_dir = './OUTPUT/Test_output/output_{}_{}/'.format(opt.model, opt.model_name)
    an = opt.angular_num
    # ------------------------------------------------------------------------#
    # Data loader
    print('===> Loading test datasets')
    test_set = DatasetFromHdf5(data_path, opt.scale)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('loaded {} LFIs from {}'.format(len(test_loader), data_path))

    test_list_file = '{}/ATO_test_lists/test_{}_list.txt'.format(opt.dataset_path, opt.test_dataset)
    fd = open(test_list_file, 'r')
    name_list = [line.strip('\n') for line in fd.readlines()]
    # -------------------------------------------------------------------------#
    # Build model
    print("===> building network")
    model = get_model(opt).to(device)
    # -------------------------------------------------------------------------#
    # test
    def ycbcr2rgb(ycbcr):
        m = np.array([[65.481, 128.553, 24.966],
                      [-37.797, -74.203, 112],
                      [112, -93.786, -18.214]])
        shape = ycbcr.shape
        if len(shape) == 3:
            ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
        rgb = copy.deepcopy(ycbcr)
        rgb[:, 0] -= 16. / 255.
        rgb[:, 1:] -= 128. / 255.
        rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
        return rgb.clip(0, 1).reshape(shape).astype(np.float32)


    def compt_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 1.0

        if mse > 1000:
            return -100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    # ------------------------------------------------------------------------#
    def test_crop():
        model.eval()
        if not os.path.exists('{}{}_results'.format(root_dir, opt.model)):
            os.makedirs('{}{}_results'.format(root_dir, opt.model))
        lf_list = []
        lf_psnr_y_list = []
        lf_ssim_y_list = []

        with torch.no_grad():
            for k, batch in enumerate(test_loader):

                lfname = name_list[k]
                print('testing LF {}-{}'.format(opt.test_dataset, lfname))


                save_dir_img = './OUTPUT/Test_output/SSR7x7/{}ximg/{}/{}'.format(opt.scale, opt.model,
                                                                                        opt.test_dataset)
                if not os.path.exists(save_dir_img):
                    os.makedirs(save_dir_img)

                angRes_in = 7
                gt_y, sr_ycbcr, Lr_SAI_y = batch[0], batch[1].numpy(), batch[2]
                gt_H = gt_y.size(-2)
                gt_W = gt_y.size(-1)
                gt_y = gt_y.numpy()
                Lr_SAI_y = Lr_SAI_y.to(device)
                Lr_SAI_y = einops.rearrange(Lr_SAI_y, 'b (c u v) h w -> b c (u h) (v w)', u=angRes_in, v=angRes_in)
                Lr_SAI_y = Lr_SAI_y.squeeze()  # (a1 h) (a2 w)
                ''' Crop LFs into Patches '''
                patch_size_for_test = 32
                stride_for_test = 16
                subLFin = LFdivide(Lr_SAI_y, angRes_in, patch_size_for_test, stride_for_test)
                numU, numV, H, W = subLFin.size()
                subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
                subLFout = torch.zeros(numU * numV, 1, angRes_in * patch_size_for_test * opt.scale, angRes_in * patch_size_for_test * opt.scale)

                ''' SR the Patches '''
                minibatch_for_test = 2
                for i in range(0, numU * numV, minibatch_for_test):
                    tmp = subLFin[i:min(i + minibatch_for_test, numU * numV), :, :, :]
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        tmp = einops.rearrange(tmp, 'b c (u h) (v w) -> b (c u v) h w', u=angRes_in, v=angRes_in)
                        out = model(tmp.to(device))
                        out = einops.rearrange(out, 'b (c u v) h w -> b c (u h) (v w)', u=angRes_in, v=angRes_in)
                        subLFout[i:min(i + minibatch_for_test, numU * numV), :, :, :] = out
                subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

                ''' Restore the Patches to LFs '''
                Sr_4D_y = LFintegrate(subLFout, angRes_in, patch_size_for_test * opt.scale,
                              stride_for_test * opt.scale, gt_H, gt_W)
                sr_y = einops.rearrange(Sr_4D_y, 'a1 a2 h w -> 1 (a1 a2) h w')
                sr_y = sr_y.cpu().numpy()

                sr_ycbcr[:, :, 0] = sr_y
                # ---------compute average PSNR/SSIM for this LFI----------#

                view_list = []
                view_psnr_y_list = []
                view_ssim_y_list = []

                for i in range(an * an):
                    height = i//an + 1
                    weight = i%an + 1
                    if opt.save_img:
                        img_save_dir = '{}/{}'.format(save_dir_img, lfname)
                        if not os.path.exists(img_save_dir):
                            os.makedirs(img_save_dir)
                        img_name = '{}/{}_{}_{}.png'.format(img_save_dir, lfname, height, weight)
                        sr_rgb_temp = ycbcr2rgb(np.transpose(sr_ycbcr[0, i], (1, 2, 0)))
                        img = (sr_rgb_temp.clip(0, 1) * 255.0).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save(img_name)

                    cur_psnr = compt_psnr(gt_y[0, i], sr_y[0, i])
                    cur_ssim = compare_ssim((gt_y[0, i] * 255.0).astype(np.uint8), (sr_y[0, i] * 255.0).astype(np.uint8),
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

                    view_list.append(i)
                    view_psnr_y_list.append(cur_psnr)
                    view_ssim_y_list.append(cur_ssim)

                lf_list.append(k)
                lf_psnr_y_list.append(np.mean(view_psnr_y_list))
                lf_ssim_y_list.append(np.mean(view_ssim_y_list))

                print(
                    'Avg. Y PSNR: {:.2f}; Avg. Y SSIM: {:.3f}'.format(np.mean(view_psnr_y_list), np.mean(view_ssim_y_list)))

        print('Over all {} LFIs on {}: Avg. Y PSNR: {:.2f}, Avg. Y SSIM: {:.3f}'.format(len(test_loader), opt.test_dataset,
                                                                                        np.mean(lf_psnr_y_list),
                                                                                        np.mean(lf_ssim_y_list)))
    # ------------------------------------------------------------------------#
    resume_path = model_dir
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'])
    print('loaded model {}'.format(resume_path))
    test_crop()

datasets = ['Kalantari', 'HCI', 'InriaSynthetic', 'Stanford_General', 'Stanford_Occlusions']

for set_name in datasets:
    print("======== Testing set: {}".format(set_name), end='')
    data_path = '{}/test_{}_x{}.h5'.format(opt.dataset_path, set_name, opt.scale)
    opt.test_dataset = set_name
    main()