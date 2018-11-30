# -*- coding: utf-8 -*-

import os
import argparse
from utils.disc import str2bool
from utils.train_tools import keras_fit

def get_parser():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--model_name',
                        help='model name for homograph',
                        default='vggstyle_homo')
    
    parser.add_argument('--patch_dataset',
                        help='patch dataset name',
                        default='cocostuff')
    
    parser.add_argument('--mask_dataset',
                        help='mask dataset name',
                        default='davis')
    
    parser.add_argument('--use_mask',
                        help='use mask with patch or not',
                        type=str2bool,
                        default=False)
    
    parser.add_argument("--batch_size",
                        help="batch size",
                        type=int,
                        default=32)
    
    parser.add_argument('--n_epoch',
                        help='training/validating epoch',
                        type=int,
                        default=100)

    parser.add_argument("--init_lr",
                        help="init learning rate",
                        type=float,
                        default=0.0001)
    
    parser.add_argument('--norm_ways',
                        help='normalize image value ways',
                        choices=['caffe','pytorch','cityscapes','-1,1','0,1'],
                        default='-1,1')
        
#    parser.add_argument("--optimizer",
#                        help="optimizer name",
#                        choices=['adam', 'sgd' ,'adamax', 'amsgrad'],
#                        default='adam')
    
    parser.add_argument('--log_dir',
                        help='base logdir for summary and checkpoints',
                        default=os.path.expanduser('~/tmp/logs'))
    
    parser.add_argument('--save_model',
                        help='save model or not',
                        type=str2bool,
                        default=True)
    
    parser.add_argument('--note',
                        help='note for summary',
                        default='000')
    
    return parser

if __name__ == '__main__':
    parser=get_parser()
    args=parser.parse_args()
    keras_fit(args)