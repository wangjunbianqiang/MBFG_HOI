import os
import time
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
# from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# from datasets.hico_constants import HicoConstants
# from datasets.hico_dataset import HicoDataset, collate_fn
# from model.vsgats.model import AGRNN
from pgception import PGception
# from model.pgception import PGception
# from model.no_frill_pose_net import fully_connect as PGception
# import utils.io as io
import cv2
import numpy as np
from utils import ConvolutionalPoseMachine, draw_body_connections, draw_keypoints, draw_masks, draw_body_box



###########################################################################################
#                                     TRAIN MODEL                                         #
###########################################################################################
def run_model(args):


    # 实例化 ConvolutionalPoseMachine 类（True 为使用预训练模型）
    estimator = ConvolutionalPoseMachine(pretrained=True)
    # opencv 读入图片
    img = cv2.imread('./HICO_test2015_00000001.jpg')
    # 传入图片至实例化后的 ConvolutionalPoseMachine 类
    pred_dict = estimator(img, masks=True, keypoints=True)
    keypointfeature = pred_dict.get('keypointrcnn').get('keypoints')
    keypointfeature_re = keypointfeature[:, :, :2]




    '''
'--batch_size', '--b_s', type=int, default=32,help='batch size: 1'
'--lr', type=float, default=0.00003, help='learning rate: 0.001'
'--d_p', type=float, default=0, help='dropout parameter: 0'  
'--bias', type=str2bool, default='true', help="add bias to fc layers or not: True"   全连接层加不加偏置，默认加上
'--bn', type=str2bool, default='false', help='use batch normailzation or not: true')  batch normailzation，默认是false
'--epoch', type=int, default=200,help='number of epochs to train: 300'
'--scheduler_step', '--s_s', type=int, default=0,help='number of epochs to train: 0'
'--o_c_l', type=list, default= [64,64,64,64],     # [64,64,128,128] [64,64,128,128] help='out channel in each branch in PGception layer: [128,256,256,256]'  每一个PGception分支的输出通道数，是一个list
'--b_l', type=int, nargs='+', default= [0,1,2,3],     # [128,256,256,256] [64,64,128,128]help='which branchs are in PGception layer: [0,1,2,3]'   
'--last_h_c', type=int, default=256,help='the channel of last hidden layer: 512'            最后一个隐藏层的输出通道数，默认是256
'--start_epoch', type=int, default=0,help='number of beginning epochs : 0'
'--c_m',  type=str, default="cat", choices=['cat', 'mean'],help='the model of last classifier: cat or mean'    默认是cat，特征融合的方式
'--optim',  type=str, default='adam', choices=['sgd', 'adam', 'amsgrad'],help='which optimizer to be use: sgd, adam, amsgrad'
'--a_n',  type=int, default=117,help='acition number: 117'               动作类别数量
'--n_layers', type=int, default=1, help='number of inception blocks: 1'     
'--agg_first', '--a_f', type=str2bool, default='true', help="In gcn, aggregation first means Z=W(AX), whilc aggregation later means Z=AWX: true"
'--attn',  type=str2bool, default='false', help="In gcn, leverage attention mechamism or not: false"
    '''
    model = PGception(action_num=args.a_n, layers=args.n_layers, classifier_mod=args.c_m, o_c_l=args.o_c_l,
                      b_l=args.b_l,
                      last_h_c=args.last_h_c, bias=args.bias, drop=args.d_p, bn=args.bn, agg_first=args.agg_first,
                      attn=args.attn)
    # model = PGception(action_num=args.a_n, drop=args.d_p)
    # load pretrained model
    # if args.pretrained:
    #     print(f"loading pretrained model {args.pretrained}")
    #     checkpoints = torch.load(args.pretrained, map_location=device)
    #     model.load_state_dict(checkpoints['state_dict'])
    device = torch.device('cuda')
    model.to(device)

    result = model(keypointfeature_re, keypointfeature_re)  # 这两个按照论文里描述的应该都是(N,17,2)的张量
    print('result', result.shape)



###########################################################################################
#                                 SET SOME ARGUMENTS                                      #
###########################################################################################
# define a string2boolean type function for argparse
def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass


parser = argparse.ArgumentParser(description="HOI DETECTION!")

# parser.add_argument('--batch_size', '--b_s', type=int, default=32,
#                     help='batch size: 1')
# parser.add_argument('--lr', type=float, default=0.00003,
#                     help='learning rate: 0.001')
parser.add_argument('--d_p', type=float, default=0,
                    help='dropout parameter: 0')
parser.add_argument('--bias', type=str2bool, default='true',
                    help="add bias to fc layers or not: True")
parser.add_argument('--bn', type=str2bool, default='false',
                    help='use batch normailzation or not: true')
# parser.add_argument('--epoch', type=int, default=200,
#                     help='number of epochs to train: 300')
# parser.add_argument('--scheduler_step', '--s_s', type=int, default=0,
#                     help='number of epochs to train: 0')
parser.add_argument('--o_c_l', type=list, default=[64, 64, 64, 64],  # [64,64,128,128] [64,64,128,128]
                    help='out channel in each branch in PGception layer: [128,256,256,256]')
parser.add_argument('--b_l', type=int, nargs='+', default=[0, 1, 2, 3],  # [128,256,256,256] [64,64,128,128]
                    help='which branchs are in PGception layer: [0,1,2,3]')
parser.add_argument('--last_h_c', type=int, default=256,
                    help='the channel of last hidden layer: 512')
# parser.add_argument('--start_epoch', type=int, default=0,
#                     help='number of beginning epochs : 0')
parser.add_argument('--c_m', type=str, default="cat", choices=['cat', 'mean'],
                    help='the model of last classifier: cat or mean')
# parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam', 'amsgrad'],
#                     help='which optimizer to be use: sgd, adam, amsgrad')
parser.add_argument('--a_n', type=int, default=117,
                    help='acition number: 117')
parser.add_argument('--n_layers', type=int, default=1,
                    help='number of inception blocks: 1')
parser.add_argument('--agg_first', '--a_f', type=str2bool, default='true',
                    help="In gcn, aggregation first means Z=W(AX), whilc aggregation later means Z=AWX: true")
parser.add_argument('--attn', type=str2bool, default='false',
                    help="In gcn, leverage attention mechamism or not: false")

# parser.add_argument('--pretrained', '-p', type=str, default=None,
#                     help='location of the pretrained model file for training: None')
# parser.add_argument('--main_pretrained', '--m_p', type=str, default='./checkpoints/hico_vsgats/hico_checkpoint.pth',
#                     help='Location of the checkpoint file of exciting method: ./checkpoints/hico_vsgats/hico_checkpoint.pth')
# parser.add_argument('--log_dir', type=str, default='./log/hico',
#                     help='path to save the log data like loss\accuracy... : ./log')
# parser.add_argument('--save_dir', type=str, default='./checkpoints/hico',
#                     help='path to save the checkpoints: ./checkpoints/vcoco')
# parser.add_argument('--exp_ver', '--e_v', type=str, default='v1',
#                     help='the version of code, will create subdir in log/ && checkpoints/ ')

# parser.add_argument('--print_every', type=int, default=10,
#                     help='number of steps for printing training and validation loss: 10')
# parser.add_argument('--save_every', type=int, default=10,
#                     help='number of steps for saving the model parameters: 50')

args = parser.parse_args()

if __name__ == "__main__":
    run_model(args)
