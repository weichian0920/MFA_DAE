# -*- coding: utf-8 -*-
#Author: Wei-Chien Wang
"""
Created on Fri Oct 16 10:24:51 2020
This code is used for unsupervised blind monoural periodic source separation based on perioidic-coded deep autoencoder (, with MSE loss)
If you find this code useful in your research, please cite:
Citation: 
       [1] K.-H. Tsai, W.-C. Wang, C.-H. Cheng, C.-Y. Tsai, J.-K. Wang, T.-H. Lin, S.-H. Fang, L.-C. Chen, and Y. Tsao, "Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder," to appear in IEEE Journal of Biomedical and Health Informatics.

Contact:
       Wei-Chien Wang
       Weichian0920@gmail.com
       Academia Sinica, Taipei, Taiwan
       
"""
import argparse
import time
from utils import misc
import torch
from torch.autograd import Variable
from datetime import datetime
import numpy as np
from model import DAE_C, DAE_F
import train
from source_separation import MFA
from dataset.HLsep_dataloader import hl_dataloader, val_dataloader
import scipy.io.wavfile as wav


# parser#

parser = argparse.ArgumentParser(description='PyTorch Source Separation')
parser.add_argument('--model_type', type=str, default='DAE_C', help='model type')
parser.add_argument('--data_feature', type=str, default='lps', help='lps or wavform')

parser.add_argument('--pretrained', default=False, help='load pretrained model or not')
parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained_model path')
parser.add_argument('--trainOrtest', type=str, default="train", help='status of training')
# training hyperparameters
parser.add_argument('--optim', type=str, default="Adam", help='optimizer for training', choices=['RMSprop', 'SGD', 'Adam'])
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=int, default=1e-4, help='initial learning rate for training (default: 1e-3)')
parser.add_argument('--CosineAnnealingWarmRestarts', type=bool, default=False, help='initial learning rate for training (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')

parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default/', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
# MFA hyperparameters
parser.add_argument('--source_num', type=str, default=3, help='number of separated sources')
parser.add_argument('--clustering_alg', type=str, default='NMF', help='clustering algorithm for embedding space')
parser.add_argument('--wienner_mask', type=bool, default=False, help='wienner time-frequency mask for output')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
misc.logger.init(args.logdir, 'train_log_')
logger = misc.logger.info

starttime = time.time()
current_time = datetime.now().strftime('%Y%m%d_%H%M')
args.logdir = args.logdir + str(args.model_type) + "_" + str(current_time)

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

"""
if args.seed is not None:
    random.seed(args.seed)
    cudnn.deterministic=None
    ngpus_per_node = torch.cuda.device_count()
"""
# build model
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
per_save_epoch = 30
t_begin = time.time()
grad_scale = args.grad_scale
# model dictionary
DAE_C_dict = {
        "frequency_bins": [0, 300],
        "encoder": [32, 16, 8],
        "decoder": [8, 16, 32, 1],
        "encoder_filter": [[1, 3], [1, 3], [1, 3]],
        "decoder_filter": [[1, 3], [1, 3], [1, 3], [1, 1]],
        "encoder_act": "relu",
        "decoder_act": "relu",
        "dense": [],
        }
DAE_F_dict = {
        "frequency_bins": [0, 300],
        "encoder": [1024, 512, 256, 128],
        "decoder": [256, 512, 1024, 1025],
        "encoder_act": "relu",
        "decoder_act": "relu",
        }

Model = {
    'DAE_C': DAE_C.autoencoder,
    'DAE_F': DAE_F.autoencoder,
}

model_dict = {
    'DAE_C': DAE_C_dict,
    'DAE_F': DAE_F_dict
}
FFT_dict = {
    'sr': 8000,
    'frequency_bins': [0, 300],
    'FFTSize': 2048,
    'Hop_length': 128,
    'Win_length': 2048,
    'normalize': True,
}
# declare model
net = Model[args.model_type](model_dict=model_dict[args.model_type], args=args, logger=logger).cuda()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    net.cuda()

if __name__ == "__main__":

    # data loader
    test_filelist = ["/home/lab001/weichian/MFA_DAE/src_pytorch/dataset/0_0.wav"]
    outdir = "{0}/test_".format(args.logdir)
    train_loader = hl_dataloader(test_filelist, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, FFT_dict=FFT_dict, args=args)
    # train
    net = train.train(train_loader, net, args, logger)
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            data = data.cuda().float()
        data = Variable(data)
        output = net(data)
        output_ = torch.reshape(output, (-1,)).detach().cpu()
    output_ = output_.numpy()
    wav.write("reconstruct.wav", 8000, np.int16(output_*32768.))

    # Source Separation by MFA analysis.
    mfa = MFA.MFA_source_separation(net, FFT_dict=FFT_dict, args=args)
    for test_file in test_filelist:
        # load test data
        lps, phase, mean, std = val_dataloader(test_file, FFT_dict)
        mfa.source_separation(np.array(lps), np.array(phase), np.array(mean), np.array(std), filedir=outdir)
