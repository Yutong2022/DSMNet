import torch.nn.functional as F
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import make_logs
import utils.utils as utility

import math
import random
import os
from collections import defaultdict

from dataset.dataset import DatasetFromHdf5_train, DatasetFromHdf5_test
from DSMNet_model.DSMNet import get_model, get_loss

from tensorboardX import SummaryWriter
import time
from lf_utils.net_utils import getNetworkDescription

import warnings

warnings.filterwarnings("ignore")

def get_cur_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# --------------------------------------------------------------------------#
# Training settings
parser = argparse.ArgumentParser(description="PyTorch LFSSR-LFIINet training")

# training settings
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=5000, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--resume_path", type=str, default="", help="resume from checkpoint path")
parser.add_argument("--pretrain", type=str, default="", help="resume from checkpoint path")
parser.add_argument("--num_cp", type=int, default=100, help="Number of epoches for saving checkpoint")
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--total_epoch", type=int, default=20000, help="Number of epoches for saving checkpoint")

# dataset
parser.add_argument("--dataset", type=str, default="all", help="Dataset for training")
parser.add_argument("--angular_num", type=int, default=7, help="Size of one angular dim")
parser.add_argument("--angRes", type=int, default=7, help="Size of one angular dim")
parser.add_argument("--angRes_in", type=int, default=7, help="Size of one angular dim")
parser.add_argument("--trainFile", type=str, default="")
parser.add_argument("--testFile", type=str, default="")

# model
parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--k_nbr", type=int, default=6, help="Size of one angular dim")

# for recording
parser.add_argument("--record", action="store_true", help="Record? --record makes it True")
parser.add_argument("--cuda", action="store_true", help="Use cuda? --cuda makes it True")
parser.add_argument("--save-dir", type=str, default="/data/liuyutong/LFSSR_DATASET_CITYU/OUTPUT/")
parser.add_argument("--num-threads", type=int, default=0)
parser.add_argument('--test_patch', action=utility.StoreAsArray, type=int, nargs='+', help="number of patches during testing")
parser.add_argument('--nf', default=48, type=int, help='')
parser.add_argument('--dataset_path', type=str, default='', help="SR model")

def main():
    global opt, model
    opt = parser.parse_args()

    opt.record = True
    opt.cuda = True
    opt.test_patch = [1, 1]

    opt.trainFile = "{}/train_all.h5".format(opt.dataset_path)
    opt.testFile = "{}/test_Kalantari_x{}.h5".format(opt.dataset_path, opt.scale)
    opt.save_dir =  './OUTPUT/'
    # --------------------------------------------------------------------------#
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
        print("Random seed is: {}".format(opt.seed))
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    opt.save_prefix = 'DSMNet_x{}_{}_{}x{}_lr{}_step{}x{}_p{}_b{}'.format(opt.scale, opt.dataset, opt.angular_num, opt.angular_num, opt.lr, opt.step, opt.reduce, opt.patch_size, opt.batch_size)

    print(opt)

    an = opt.angular_num
    # --------------------------------------------------------------------------#
    # Data loader
    print('===> Loading train datasets')
    train_set = DatasetFromHdf5_train(opt.trainFile, opt.scale, opt.patch_size)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)
    print('loaded {} LFIs from {}'.format(len(train_loader), opt.trainFile))
    print('===> Loading test datasets')
    test_set = DatasetFromHdf5_test(opt.testFile, opt.scale)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=opt.num_threads)
    print('loaded {} LFIs from {}'.format(len(test_loader), opt.testFile))

    # --------------------------------------------------------------------------#
    # Build model
    print("===> building network")
    model = get_model(opt)

    # criterion = nn.L1Loss()
    criterion = get_loss(opt)

    if opt.cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    # -------------------------------------------------------------------------#
    # optimizer and loss logger
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)

    # ------------------------------------------------------------------------#
    # optionally use a pretrained model for initialization
    if opt.pretrain:
        if os.path.isfile(opt.pretrain):
            print('===> pretrain model is {}'.format(opt.pretrain))
            ckpt = torch.load(opt.pretrain)
    else:
        print('No pretrain model, initialize randomly')

    # ------------------------------------------------------------------------#
    # optionally resume from a checkpoint
    if opt.resume_path:
        resume_path = opt.resume_path
        if os.path.isfile(resume_path):
            print("==> loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            opt.resume_epoch = checkpoint['epoch']
            losslogger = checkpoint['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))
            opt.resume_epoch = 0
    else:
        opt.resume_epoch = 0

    # ------------------------------------------------------------------------#
    print('==> training')
    if opt.record:
        make_logs("{}log/{}/".format(opt.save_dir, opt.save_prefix), "train_log.log", "train_err.log")
        writer = SummaryWriter(log_dir="{}logs/{}".format(opt.save_dir, opt.save_prefix),
                               comment="Training curve for LFASR-SAS-2-to-8")
        if not os.path.exists("{}checkpoints/{}/".format(opt.save_dir, opt.save_prefix)):
            os.makedirs("{}checkpoints/{}/".format(opt.save_dir, opt.save_prefix))

    model = nn.DataParallel(model)
    for epoch in range(opt.resume_epoch + 1, opt.total_epoch):
        loss = train(epoch, model, scheduler, train_loader, optimizer, losslogger, criterion)

        if epoch % opt.num_cp == 0 or epoch > opt.total_epoch - 100:
            model_save_path = os.path.join("{}checkpoints/{}/model_epoch_{}.pth".format(opt.save_dir, opt.save_prefix, epoch))
            state = {'epoch': epoch, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

        if opt.record:
            writer.add_scalar("train/recon_loss", loss, epoch)

# -----------------------------------------------------------------------#

def train(epoch, model, scheduler, train_loader, optimizer, losslogger, criterion):
    model.train()
    scheduler.step()

    print("{}: Epoch = {}, lr = {}".format(get_cur_time(), epoch, optimizer.param_groups[0]["lr"]))

    loss_count = 0.

    for i, batch in enumerate(train_loader, 1):

        lr = batch[int(math.log(opt.scale, 2))].cuda()
        hr = batch[0].cuda()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss_count += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{}: Epoch {}, [{}/{}]: SR loss: {:.10f}".format(get_cur_time(), epoch, i, len(train_loader),
                                                               loss.cpu().data))

    average_loss = loss_count / len(train_loader)
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(average_loss)

    return average_loss

if __name__ == '__main__':
    main()


