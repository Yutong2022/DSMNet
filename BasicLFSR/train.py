from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
import torch.nn as nn

def main(args):
    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'DSMNet_model.' + 'DSMNet'
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)
    net = net.to(device)

    if args.MGPU == 4:
        net = nn.DataParallel(net, device_ids=[0,1,2,3])
    elif args.MGPU == 6:
        net = nn.DataParallel(net, device_ids=[0,1,2,3,4,5])
    elif args.MGPU == 2:
        net = nn.DataParallel(net, device_ids=[0,1])
    elif args.MGPU == 1:
        net = nn.DataParallel(net)
    else:
        print('args setting errors may occur')

    # ------------------------------------------------------------------------#
    # Fineturn using a pretrained model for initialization
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print('===> pretrain model is {}'.format(args.pretrain))
            ckpt = torch.load(args.pretrain)
            net.load_state_dict(ckpt['model'])
        else:
            print('===> no model found at {}'.format(args.pretrain))
    else:
        print('No pretrain model, this is a new train model')    
    # ------------------------------------------------------------------------#
    
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)


    ''' optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    # ------------------------------------------------------------------------#
    # Resume from a checkpoint
    if args.resume_path:
        resume_path = args.resume_path
        if os.path.isfile(resume_path):
            print("==> loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.resume_epoch = checkpoint['epoch'] + 1
        else:
            print("==> no model found at 'epoch{}'".format(args.resume_epoch))
            args.resume_epoch = 0
    else:
        args.resume_epoch = 0
        print('No resume model, this train is from epoch 0')   
    # ------------------------------------------------------------------------#

    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(args.resume_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(train_loader, device, net, criterion, optimizer)
        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))


        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
            args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
            state = {'epoch': idx_epoch, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))


        ''' Validation '''
        step = 1
        if (idx_epoch + 1)%step==0 or idx_epoch > args.epoch-step:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                # excel_file = ExcelFile()

                psnr_testset = []
                ssim_testset = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
                    # excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)


                    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
                    ssim_epoch_test = float(np.array(ssim_iter_test).mean())


                    psnr_testset.append(psnr_epoch_test)
                    ssim_testset.append(ssim_epoch_test)
                    logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
                    idx_epoch + 1, test_name, psnr_epoch_test, ssim_epoch_test))
                    pass
                psnr_mean_test = float(np.array(psnr_testset).mean())
                ssim_mean_test = float(np.array(ssim_testset).mean())
                logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
                                  % (psnr_mean_test, ssim_mean_test))
                # excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')
                pass
            pass

        ''' scheduler '''
        scheduler.step()
        pass
    pass


def train(train_loader, device, net, criterion, optimizer):
    ''' training one epoch '''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device)      # low resolution
        label = label.to(device)    # high resolution
        out = net(data, data_info)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())
        psnr, ssim = cal_metrics(args, label, out)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train


def test_wopad(test_loader, device, net, save_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        Hr_SAI_y = Hr_SAI_y
        Sr_SAI_cbcr = Sr_SAI_cbcr

        ''' Crop LFs into Patches '''
        subLF_main, subLF_left, subLF_down, subLF_slope = LFdivide1(Lr_SAI_y, args.angRes_in, args.patch_size_for_test)
    
        numU, numV, H, W = subLF_main.size()
        subLF_main = rearrange(subLF_main, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        subLF_left = rearrange(subLF_left, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        subLF_down = rearrange(subLF_down, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        subLF_slope= rearrange(subLF_slope, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        minibatch1 = 1
        minibatch2 = 1
        minibatch3 = 1
        SR_main = torch.zeros(numU * numV, 1, H * args.scale_factor, W * args.scale_factor)
        SR_left = torch.zeros(numV, 1, H * args.scale_factor, W * args.scale_factor)
        SR_down = torch.zeros(numU, 1, H * args.scale_factor, W * args.scale_factor)
        SR_slope = torch.zeros(1, 1, H * args.scale_factor, W * args.scale_factor)

        for i in range(0, numU * numV, minibatch1):
            tmp = subLF_main[:min(i + minibatch1, numU * numV), :, :, :]
            with torch.no_grad():
                    net.eval()
                    tmp = net(tmp.to(device), data_info)
                    torch.cuda.empty_cache()
                    SR_main[:min(i + minibatch1, numU * numV), :, :, :] = tmp

        for i in range(0, numV, minibatch2):
            tmp = subLF_left[:min(i + minibatch2, numV), :, :, :]
            with torch.no_grad():
                    net.eval()
                    tmp = net(tmp.to(device), data_info)
                    torch.cuda.empty_cache()
                    SR_left[:min(i + minibatch2, numV), :, :, :] = tmp

        for i in range(0, numU, minibatch3):
            tmp = subLF_down[:min(i + minibatch3, numU), :, :, :]
            with torch.no_grad():
                    net.eval()
                    tmp = net(tmp.to(device), data_info)
                    torch.cuda.empty_cache()
                    SR_down[:min(i + minibatch3, numU), :, :, :] = tmp

        with torch.no_grad():
                net.eval()
                tmp = net(subLF_slope.to(device), data_info)
                torch.cuda.empty_cache()
                SR_slope[:,:,:,:] = tmp

        SR_main = rearrange(SR_main, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU ,n2=numV)
        SR_left = rearrange(SR_left, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=1 ,n2=numV)
        SR_down = rearrange(SR_down, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU ,n2=1)
        SR_slope = rearrange(SR_slope, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=1 ,n2=1)

        LFout = torch.zeros_like(Hr_SAI_y)
        LFout = rearrange(LFout, '1 1 (u h) (v w) -> (u v) h w', u=args.angRes_out, v=args.angRes_out)
        Sr_4D_y = LFintegrate1(SR_main, SR_left, SR_down, SR_slope, args.angRes_out, args.patch_size_for_test, LFout, args.scale_factor)
        Sr_SAI_y = rearrange(Sr_4D_y, '(u v) h w -> 1 1 (u h) (v w)', u=args.angRes_out, v=args.angRes_out)
        # Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        LF_iter_test.append(LF_name[0])

        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            views_dir = save_dir_.joinpath('views')
            views_dir.mkdir(exist_ok=True)
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            img = Sr_4D_rgb[args.angRes_out // 2, args.angRes_out // 2, :, :, :]
            path = str(save_dir_) + '/' + LF_name[0] + '_' + 'CenterView.bmp'
            imageio.imwrite(path, img)
            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)
                    pass
                pass
            pass
        pass

    return psnr_iter_test, ssim_iter_test, LF_iter_test

def test(test_loader, device, net, save_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        Hr_SAI_y = Hr_SAI_y
        Sr_SAI_cbcr = Sr_SAI_cbcr

        ''' Crop LFs into Patches '''
        subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        
        subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                               args.angRes_in * args.patch_size_for_test * args.scale_factor)

        ''' SR the Patches '''
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            with torch.no_grad():
                net.eval()
                # torch.cuda.empty_cache()
                out = net(tmp.to(device), data_info)
                subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
        subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

        ''' Restore the Patches to LFs '''
        Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes_out, Hr_SAI_y.size(-1)//args.angRes_out)
        Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        LF_iter_test.append(LF_name[0])


        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            views_dir = save_dir_.joinpath('views')
            views_dir.mkdir(exist_ok=True)
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            img = Sr_4D_rgb[args.angRes_out // 2, args.angRes_out // 2, :, :, :]
            path = str(save_dir_) + '/' + LF_name[0] + '_' + 'CenterView.bmp'
            imageio.imwrite(path, img)
            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)
                    pass
                pass
            pass
        pass

    return psnr_iter_test, ssim_iter_test, LF_iter_test

if __name__ == '__main__':
    from option import args

    main(args)
