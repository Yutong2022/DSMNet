import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import MultiTestSetDataLoader
from collections import OrderedDict
from train import test, test_wopad
import torch.nn as nn

def main(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)

    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'DSMNet_model.' + 'DSMNet'
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)
    net = net.to(device)
    net = nn.DataParallel(net)

    ''' Load Pre-Trained PTH '''
    print('===> pretrain model is {}'.format(args.path_pre_pth))
    ckpt = torch.load(args.path_pre_pth)
    net.load_state_dict(ckpt['model'])

    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''

        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)
            if test_name in ['EPFL', 'INRIA_Lytro']:
                # The operation 'padding' will introduce noise to our DSMNet in some scenes under the scale of 4.
                psnr_iter_test, ssim_iter_test, LF_name = test_wopad(test_loader, device, net, save_dir)
            else:
                psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)

            psnr_epoch_test = float(np.array(psnr_iter_test).mean())
            ssim_epoch_test = float(np.array(ssim_iter_test).mean())
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print('Test on %s, psnr/ssim is %.2f/%.3f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass

        psnr_mean_test = float(np.array(psnr_testset).mean())
        ssim_mean_test = float(np.array(ssim_testset).mean())
        print('The mean psnr on testsets is %.5f, mean ssim is %.5f' % (psnr_mean_test, ssim_mean_test))

    pass


if __name__ == '__main__':
    from option import args

    main(args)
