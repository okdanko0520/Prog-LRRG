import torch
import os
from torch import Tensor, device
import datetime
from dateutil import tz
import yaml
import random
import matplotlib.pyplot as plt
import threading
import json
import pytorch_lightning as pl
import numpy as np
from dataset.dataloaders import DataLoaders
from dataset.datasets import BaseDataset
from modules.tokenizers import Tokenizer
import argparse
from transformers import GPT2TokenizerFast
from model.model_diff import Pretrain
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

def enumerated_save_path(save_dir, save_name, extension):
    save_path = os.path.join(save_dir, save_name + extension)
    assert '.' in extension, 'No period in extension.'
    if os.path.isfile(save_path):
        count = 2
        while True:
            save_path = os.path.join(save_dir, save_name + "_" + str(count) + extension)
            count += 1
            if not os.path.isfile(save_path):
                break

    return save_path

def str2bool(value):
    if value.lower() in ['yes', 'true', 't', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_agrs():
    parser = argparse.ArgumentParser()

    # basic configuration
    parser.add_argument('--task', type=str, default='test',choices=['pretrain', 'finetune', 'test'])

    # Data input settings
    parser.add_argument('--images_dir', type=str, default='/media/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files', help='the path to the directory containing the data.')
    parser.add_argument('--data_name', type=str, choices=['mimic_cxr', 'mimic_abn', 'twoview_cxr'], default='mimic_cxr')
    parser.add_argument('--ann_path', type=str, default='/media/fengxinru/CheXRel_global_report/dataset/mimic_cxr_annotation_sen.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=128, help='the maximum sequence length of the reports.')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--is_save_checkpoint', type=str2bool, default='yes', help='whether save checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')
    parser.add_argument('--multiview_fusion_num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--text_encoder_num_layers', type=int, default=6)
    parser.add_argument('--is_prior_report', type=str2bool, default='yes', help='whether using prior report for finetune')
    parser.add_argument('--is_multiview_fusion', type=str2bool, default='yes', help='whether using multiple positive contrastive learning')
    parser.add_argument('--using_local_loss', type=str2bool, default='yes', help='whether token-wise cross-modal alignment loss')
    parser.add_argument('--using_diff_loss', type=str2bool, default='yes', help='whether multi-positive contrastive loss for pretraining')
    parser.add_argument('--ckpt_zoo_dir', type=str,
                       default='/media/ext_mm/fdd/CheXRel_global_report/checkpoints',
                       help='if using local checkpoint, this variable must be provided')
    parser.add_argument('--report_style', type=str, choices=['report', 'factual_serialization'],
                       default='factual_serialization', help='the style of reports for cross-modal alignment')
    parser.add_argument('--is_indication', type=str2bool, default='yes', help='whether using indication')
    parser.add_argument('--candi', type=str, default=None, help='Candidate checkpoint path')
    parser.add_argument('--cross_modal_fusion_num_layers', type=int, default=1)
    # ========= metrics checkpoint config =====#
    #需要改一下地址fix
    parser.add_argument('--vit_path', type=str, default='microsoft/rad-dino', help='checkpoint')
    parser.add_argument('--chexbert_path', type=str, default='/media/fengxinru/CheXRel_global_report/ext/chexbert/chexbert.pth', help='checkpoint')
    parser.add_argument('--vit_ChexRel_path', type=str, default='/media/fengxinru/CheXRel_global_report/checkpoints /best_ckpt.pt', help='checkpoint')
    parser.add_argument('--bert_path', type=str, default='/media/fengxinru/CheXRel_global_report/ext/bert-base-uncased', help='checkpoint')
    parser.add_argument('--rad_dino_path', type=str, default='/media/fengxinru/CheXRel_global_report/ext/rad_dino', help='checkpoint')
    # ========= backbone checkpoint config =====#
    parser.add_argument('--cxr_bert_path', type=str, default='/media/fengxinru/CheXRel_global_report/ext/BiomedVLP-CXR-BERT', help='checkpoint')
    # TODO:这里我放的是4090上的路径,在3090上跑请修改。
    parser.add_argument('--distilgpt2_path', type=str, default='/media/fengxinru/CheXRel_global_report/ext/distilgpt2', help='checkpoint')
    parser.add_argument('--cvt2distilgpt2_path', type=str,
                    #    default='/media/fengxinru/CheXRel_global_report/scirpt/results/mimic_cxr/finetune/v0623-MLRG-ft-RCB_2025_06_24_01/checkpoint/last.ckpt',
                       help='baseline checkpoint')
    parser.add_argument('--radgraph_path', type=str, default='/media/fengxinru/CheXRel_global_report/ext/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz', help='checkpoint')
    # Trainer settings
    parser.add_argument('--pt_lr', type=float, default=5.0e-6)  # 5.0e-5
    parser.add_argument('--ft_lr', type=float, default=5.0e-5)  # 5.0e-5
    parser.add_argument('--monitor_metric', type=str, default='RCB',help='the metric is used to selecting best models. pretraining is all_loss, while fine-tuning is RCB')
    # choices={all_metrics, metrics, RC, RB, RCB}
    parser.add_argument('--epochs', type=int, default=50, help='the number of training epochs.')
    parser.add_argument('--record_dir', type=str, default='/media/fengxinru/CheXRel_global_report/records', help='the patch to save the results of experiments')
    parser.add_argument('--hidden_size', type=int, default=768, help='the dimension of unify embedding for image and text features')
    # 需要调整参数
    parser.add_argument('--temp', type=float, default=0.5, help='temperature parameter for instance-wise cross-modal alignment')  # 5.0e-5
    parser.add_argument('--region_temp', type=float, default=0.5, help='temperature parameter for instance-wise cross-modal alignment')  # 5.0e-5
    parser.add_argument('--diff_temp', type=float, default=0.07, help='temperature parameter for instance-wise cross-modal alignment')  # 5.0e-5
    parser.add_argument('--load', type=str, help='whether to load the pre-trained model.',
                       # default='script/results/mimic_cxr/pretrain/v0906_fs_pt_2024_09_06_12/checkpoint/best_model.ckpt'
                       # default='script/results/mimic_cxr/finetune/v0915_ft-fs_2024_09_17_10/checkpoint/last-1.0329.ckpt'
                       )
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.',
                       default='/media/fengxinru/CheXRel_global_report/scirpt/results/mimic_cxr/finetune/v0827-MLRG-ft-RCB_2025_08_27_09/checkpoint/last.ckpt'
                       )
    parser.add_argument('--test_ckpt_path', type=str, help='checkpoint for test',
                       default='/home/miao/data/Code/five_gpt2/script/results/mimic_cxr/finetune/v1011-MLRG-ft-RCB_2024_10_12_14/best_model.ckpt',
                       )
    parser.add_argument('--version', type=str, default='long_sentence', help='the name of experiment')
    
    # parser.add_argument('--load', type=str, default='script/results/mimic_cxr/pretrain/v0906_fs_pt_2024_09_06_12/checkpoint/best_model.ckpt', help='whether to load the pre-trained model.')

    # implementation config
    parser.add_argument('--seed', type=int, default=9233, help='random seed')
    parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
    parser.add_argument('--num_beams', type=int, default=3, help='beam size for language generation')

    # record results
    parser.add_argument('--exp_dir_trial', type=str, default='results',help='fold path for recording experimental results')
    parser.add_argument('--print_step', type=int, default=500, help='the frequency of print')
    # optimizer settings
    parser.add_argument('--lr', default=0.01, type=float)

    # moedel
    parser.add_argument('--pretrain', default=None, type=str) # 预训练权重路径

    

    
    # finish
    # args, _ = parser.parse_known_args()
    args = parser.parse_args() # pretrain / finetune     
    # args = parser.parse_args(args=[]) #创建数据集
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H")
    # args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.exp_dir_trial = f'{args.exp_dir_trial}/{args.data_name}/{args.task}/{args.version}_{extension}'
    os.makedirs(args.exp_dir_trial, exist_ok=True)

    # config logger
    logger = SetLogger(f'{args.exp_dir_trial}/log_{extension}.log','a')

    # determine absolute path for checkpoints
    # if not args['online_checkpoint']:
    candi_list = ['chexbert_path', 'radgraph_path', "bert_path", "cxr_bert_path",
                  "cvt2distilgpt2_path", "distilgpt2_path", "rad_dino_path"]
    for candi in candi_list:
        if args.candi is None:
            continue
        args.candi = os.path.join(args.ckpt_zoo_dir, args.candi)

    # determine the monitor_mode
    args.monitor_mode = 'max'
    if args.task == 'pretrain':  # pretrain
        args.monitor_mode = 'min'
        args.monitor_metric = 'val_epoch_loss'
    checkpoint_dir = os.path.join(args.exp_dir_trial, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    args.time = extension

    # save parameters
    config_dir = f"{args.exp_dir_trial}/configs"
    os.makedirs(config_dir, exist_ok=True)
    file_name = f"{config_dir}/config_{extension}.yaml"
    print(f'parameters is saved in {file_name}')
    with open(file_name, 'w') as file:
        yaml.dump(args, file, default_flow_style=False)

    return args, logger
    # return args, None


def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class SetLogger:
    def __init__(self, filepath, mode='a', lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi-process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            raise ValueError("Mode must be 'w' or 'a'")
        self.mode = mode
        self.lock = lock or threading.Lock()

        try:
            self.file = open(self.filepath, self.mode)
        except Exception as e:
            print(f"Failed to open log file: {e}")
            raise
    
    def info(self, message):
        """
        Log an info message to the file.
        :param message: The message to log
        """
        with self.lock:
            try:
                self.file.write(message + '\n')
                self.file.flush()
            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def __del__(self):
        """Ensure that the file is closed when the logger is destroyed."""
        try:
            if not self.file.closed:
                self.file.close()
        except Exception as e:
            print(f"Failed to close log file: {e}")

