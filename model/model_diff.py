"""
lcxr+mrlg+chexrelformer修改
image-text模态对齐
"""
# 包载入
import torch
import torch.nn as nn
import numpy as np
import json
import math
import time
import torchmetrics
from typing import Dict
import torch
from torch import nn
import transformers
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2TokenizerFast, AutoModel, AutoConfig, AutoImageProcessor
from transformers.configuration_utils import PretrainedConfig
from modules.CheXRelVisualExt import CheXRelVisualExtracter
from model.bert_model import Transformer, BertCrossLayer
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.chexbert import RadGraphMetrics, F1CheXbertMetrics
from tools.metrics.report_logger import ReportLogger

from ext.resnet import ProjectionHead, get_extended_attention_mask
from modules.CheXRelFormer import CheXRelFormer_FeatureExtractor
from modules.CheXRelFormer_onlydiff import CheXRelFormer_FeatureExtractor
from dataset.datasets import BaseDataset, ReportDataset
from dataset.dataloaders import DataLoaders
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

class Pretrain(pl.LightningModule):
    def __init__(self, args, tokenizer: GPT2TokenizerFast, logger, **kwargs):
        super(Pretrain, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        print(f"loggr: {logger}")
        self.mylog = logger
        # 保留确定性设置，但是只警告，不报错
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.train_set = None
        self.val_set = None
        self.test_set = None
        # 每个工作线程会预先获得5个数据
        self.prefetch_factor = 5

        self.val_min_losses = {
            "epoch": -1, # 记录在哪个 epoch 取得了最小的验证损失。
            "mpc_loss": 1000, # 记录验证集中 mpc_loss
            "instance_loss": 1000, # 记录验证集中的 instance_loss
            "sen_text_loss": 1000, # 记录验证集中的 sen_text_loss
            "diff_loss": 1000, # 记录验证集中的 diff_loss
            'loss': 1000 # 记录最小的损失值
        }  # loss, mpc_loss, instance_loss

        self.train_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(args.device),
            'mpc_loss': torchmetrics.MeanMetric().to(args.device),
            'instance_loss': torchmetrics.MeanMetric().to(args.device),
            'sen_text_loss': torchmetrics.MeanMetric().to(args.device),
            'diff_loss': torchmetrics.MeanMetric().to(args.device),
        }
        self.val_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(args.device),
            'mpc_loss': torchmetrics.MeanMetric().to(args.device),
            'instance_loss': torchmetrics.MeanMetric().to(args.device),
            'sen_text_loss': torchmetrics.MeanMetric().to(args.device),
            'diff_loss': torchmetrics.MeanMetric().to(args.device),
            
        }
        self.test_loss_metric = {
            'loss': torchmetrics.MeanMetric(),
            'mpc_loss': torchmetrics.MeanMetric(),
            'instance_loss': torchmetrics.MeanMetric(),
            'sen_text_loss': torchmetrics.MeanMetric(),
            'diff_loss': torchmetrics.MeanMetric(),
        }


        # Image Diff Encoder1
        # 可以直接从CheXRelFormer_FeatureExtractor提取到特征，只需要写一个接口传入就可以了
        # self.image_encoder_1 = CheXRelFormer_FeatureExtractor().to(args.device)
        # # 因为图像编码器是我自己定义的，我知道隐藏层维度，直接就写死了
        # try:
        #     image_dim = self.image_encoder_1.config.hidden_size
        #     self.image_encoder_1.eval()
        #     for param in self.image_encoder_1.parameters():
        #         param.requires_grad = False
        # except AttributeError:
        #     # 如果 config 或 hidden_size 不存在，回退到默认值
        #     image_dim = 768
        self.image_encoder_1 = CheXRelVisualExtracter(args.vit_ChexRel_path).to(args.device)
        # 因为图像编码器是我自己定义的，我知道隐藏层维度，直接就写死了
        try:
            image_dim = self.image_encoder_1.config.hidden_size
            self.image_encoder.eval_1()
            for param in self.image_encoder_1.parameters():
                param.requires_grad = False
        except AttributeError:
            # 如果 config 或 hidden_size 不存在，回退到默认值
            image_dim = 768
            self.image_encoder_1.train()
            for param in self.image_encoder_1.parameters():
                param.requires_grad = True
            print("是否可以不用设置自己评估！")
            print(self.image_encoder_1 in self.children())
        
        #  Image Local & Global Encoder2
        self.image_processor = AutoImageProcessor.from_pretrained(args.rad_dino_path)
        self.image_encoder_2 = AutoModel.from_pretrained(args.rad_dino_path)
        # 获取图像编码器的隐藏层为度，在后续的特征投影或者融合中使用
        image_dim_2 = self.image_encoder_2.config.hidden_size
        self.image_encoder_2.eval()
        for param in self.image_encoder_2.parameters():
            param.requires_grad = False

        # Text Encoder
        self.text_encoder = self.build_text_encoder()
        # 该维度会在后续的特征投影或融合中使用，将文本特征投影到与图像特征相同的空间。
        text_dim = self.text_encoder.config.hidden_size
        # 小火苗
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # projection head-用于跨模态对齐
        # 影像投影
        self.image_projection_1 = ProjectionHead(image_dim, args.hidden_size * 2, args.hidden_size)
        self.image_projection_2 = ProjectionHead(image_dim_2, args.hidden_size * 2, args.hidden_size)
        # 文本投影，于将文本编码器生成的高维特征（text_dim）投影到与图像特征相同的特征空间。
        self.text_projection = ProjectionHead(text_dim, args.hidden_size* 2, args.hidden_size)

        # layer_norm--归一化
        self.ln_1 = nn.LayerNorm(image_dim_2)
        self.ln_2 = nn.LayerNorm(args.hidden_size)

        # fusion cur and prior module
        self.fusion_multiview = Transformer(args.hidden_size, args.multiview_fusion_num_layers,
                                            heads=args.num_heads,
                                            dim_head=args.hidden_size // 4,
                                            mlp_dim=args.hidden_size)

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args.cxr_bert_path, trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args.text_encoder_num_layers
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args.cxr_bert_path,
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)
    
    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = ReportDataset(self.args, self.tokenizer, split='train',img_size=256, is_train=True)
            self.val_set = ReportDataset(self.args, self.tokenizer, split='val',img_size=256, is_train=False)
            # print(
            #     "No. of training & validation examples: {} & {}.".format(
            #         self.train_set.__len__(), self.val_set.__len__()
            #     )
            # )
            # self.mylog.info("No. of training & validation examples: {} & {}.".format(
            #     self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:  # fit
            self.test_set = ReportDataset(self.args, self.tokenizer, split='test',img_size=256, is_train=False)
            # print("No. of test examples: {}.".format(self.test_set.__len__()))
            # self.mylog.info("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        collate_fn = DataLoaders.pretrain_collate_fn
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        collate_fn = DataLoaders.pretrain_collate_fn
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        collate_fn = DataLoaders.pretrain_collate_fn
        return DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )
    
    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        if self.args.task == 'pretrain':
            optimiser = torch.optim.AdamW(self.parameters(), lr=self.args.pt_lr)
            lr_scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.args.monitor_metric,
                    'frequency': 1   # the frequency of check
                }
            }
        else:
            pretrain_main_params, finetune_mai_params = [], []
            if self.args.load is not None:
                checkpoint = torch.load(self.args.load)['state_dict']
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in checkpoint:
                        pretrain_main_params.append(param)
                    else:
                        finetune_main_params.append(param)
            else:
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    finetune_main_params.append(param)

            optimiser = torch.optim.AdamW(
                [{'params': pretrain_main_params, 'lr': self.args.pt_lr},
                 {'params': finetune_main_params, 'lr': self.args.ft_lr}])

            lr_scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.args.monitor_metric,
                    'frequency': 1   # the frequency of check
                }
            }

    def tokenization(self, text, pair_text=None, device=None):
        if pair_text is None:
            inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=True,
                                    max_length=self.args.max_length, truncation=True)
                                    # max_length=self.args['max_length'], truncation=True)
        else:
            inputs = self.tokenizer(text, pair_text, padding=True, return_token_type_ids=True,
                                    return_tensors='pt', max_length=200, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        return inputs
    
    # Notice：调试注意这里
    def multiview_fusion_network(self, cur_image_embed, prev_image_embed , batch_size):
        # obtain labels indicate corresponding multiview images
        # 只融合历史影像
        new_image_embed = []
        for i in range(batch_size):
            # include multiview images
            image_embed = self.fusion_multiview(cur_image_embed[i], prev_image_embed[i],
                                                prev_image_embed[i])

            new_image_embed.append(image_embed)
        new_image_embed = torch.stack(new_image_embed, dim=0)
        return new_image_embed

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        # obtain multi-positive target
        # 对 patient_ids 进行裁剪，使其长度与 global_image_embed 的第 0 维（即样本数量）保持一致。
        # print('global_image_embed shape: ', global_image_embed.shape)
        patient_ids = patient_ids[:global_image_embed.shape[0]]
        # print('patient_ids: ', patient_ids)
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels).float().to(global_image_embed.device)
        labels = labels / labels.sum(1, keepdim=True)
        # print(f"labels: {labels.shape}, labels: {labels}")
        del patient_ids

        # normalize
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2) # [B, 768]
        global_text_embed = F.normalize(global_text_embed, dim=-1, p=2) # [B, 768]
        # calculate the InfoNCE loss
        instance_sim = global_image_embed @ global_text_embed.t()
        instance_sim_1 = global_text_embed @ global_image_embed.t()

        #  # === 添加相似度监控 ===
        # with torch.no_grad():
        #     sim_min = instance_sim.min().item()
        #     sim_max = instance_sim.max().item()
        #     sim_mean = instance_sim.mean().item()
        #     sim_std = instance_sim.std().item()
        #     self.mylog.info(
        #         f"Similarity Stats - Min: {sim_min:.4f}, Max: {sim_max:.4f}, Mean: {sim_mean:.4f}, Std: {sim_std:.4f}"
        #     )
            
        # temp控制相似度分布的尖锐程度
        loss_instance_1 = F.cross_entropy(instance_sim / self.args.temp, labels)
        loss_instance_2 = F.cross_entropy(instance_sim_1 / self.args.temp, labels)
        # print(f"\nloss_instance_1: {loss_instance_1}, loss_instance_2: {loss_instance_2}\n")
        global_instance_loss = (loss_instance_1 + loss_instance_2) / 2.0
        return global_instance_loss

    def local_text_token_alignment_loss(self, local_image_embed, local_text_embed):
        # cross-modal alignment between image patches and sentence embed in reports

        t_att_sim = local_text_embed @ local_image_embed.permute(0, 2, 1).contiguous()
        t_att_sco = F.softmax(t_att_sim / math.sqrt(local_image_embed.shape[2]), dim=-1)
        t_att_output = torch.bmm(t_att_sco, local_image_embed)

        device = local_image_embed.device
        # normalize
        t_att_output = F.normalize(t_att_output, dim=-1, p=2)
        local_text_embed = F.normalize(local_text_embed, dim=-1, p=2)
        # calculate the loss
        word_sim = torch.bmm(local_text_embed, t_att_output.permute(0, 2, 1).contiguous()) / self.args.region_temp
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        loss_word = (loss_word_2 + loss_word_1) / 2.0
        return loss_word
    
    # 差异对比学习模块
    def diff_alignment_loss(self, diff_image_embed, diff_text_embed, is_aligned):
        """
        对齐图像差异特征和文本差异特征的对比学习损失。
        """
        # 确保 is_aligned 是张量
        if isinstance(is_aligned, list):
            is_aligned = torch.tensor(is_aligned, device=diff_text_embed.device)

        # 过滤出需要对齐的样本
        aligned_diff_image_embed = diff_image_embed[is_aligned]
        aligned_diff_text_embed = diff_text_embed[is_aligned]

        # 如果没有需要对齐的样本，返回零损失
        if aligned_diff_image_embed.size(0) == 0:
            return torch.tensor(0.0, device=diff_image_embed.device)
        
        # print(f"测试aligned_diff_image_embed: {aligned_diff_image_embed.shape}, aligned_diff_text_embed: {aligned_diff_text_embed.shape}")
        aligned_diff_image_embed = aligned_diff_image_embed.mean(dim=1)  # 形状变为 [15, 768]

        # 归一化特征
        aligned_diff_image_embed = F.normalize(aligned_diff_image_embed, dim=-1, p=2)
        aligned_diff_text_embed = F.normalize(aligned_diff_text_embed, dim=-1, p=2)

        # 计算相似度矩阵
        diff_sim = aligned_diff_image_embed @ aligned_diff_text_embed.t()
        diff_sim_1 = aligned_diff_text_embed @ aligned_diff_image_embed.t()

        # 构造标签
        labels = torch.arange(diff_sim.size(0)).to(diff_sim.device)
        
        # 计算 InfoNCE 损失
        loss_diff_1 = F.cross_entropy(diff_sim / self.args.diff_temp, labels)
        loss_diff_2 = F.cross_entropy(diff_sim_1 / self.args.diff_temp, labels)
        diff_loss = (loss_diff_1 + loss_diff_2) / 2.0

        return diff_loss

        
    # TODO:对齐的文本变化特征应该怎么设计更好呢？最后决定使用文本的局部特征进行对齐
    # def diff_alignment_loss(self, diff_image_embed, diff_text_embed, is_aligned):
    #     # 确保 is_aligned 是张量
    #     if isinstance(is_aligned, list):
    #         is_aligned = torch.tensor(is_aligned, device=diff_text_embed.device)

    #     # 过滤出需要对齐的样本
    #     aligned_diff_image_embed = diff_image_embed[is_aligned]
    #     aligned_diff_text_embed = diff_text_embed[is_aligned]

    #     # 如果没有需要对齐的样本，返回零损失
    #     if aligned_diff_image_embed.size(0) == 0:
    #         return torch.tensor(0.0, device=diff_image_embed.device)
        
    #      # 仅对非空样本进行计算
    #     diff_image_embed = aligned_diff_image_embed
    #     diff_text_embed = aligned_diff_text_embed
        
    #     # cross-modal alignment between diff_image patches and sentence embed in reports
    #     t_att_sim = diff_text_embed @ diff_image_embed.permute(0, 2, 1).contiguous()
    #     t_att_sco = F.softmax(t_att_sim / math.sqrt(diff_image_embed.shape[2]), dim=-1)
    #     t_att_output = torch.bmm(t_att_sco, diff_image_embed)

    #     device = diff_image_embed.device
    #     # normalize
    #     t_att_output = F.normalize(t_att_output, dim=-1, p=2)
    #     diff_text_embed = F.normalize(diff_text_embed, dim=-1, p=2)
    #     # calculate the loss
    #     word_sim = torch.bmm(diff_text_embed, t_att_output.permute(0, 2, 1).contiguous()) / self.args.region_temp
    #     word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
    #     word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
    #     loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

    #     word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
    #     loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
    #     diff_loss = (loss_word_2 + loss_word_1) / 2.0

    #     return diff_loss
    
    # 需要调整（已调整）如果效果不好的，尝试使用raddino图像处理器
    # def encoder_forward(self, images_cur, images_prev, inputs, comprisions):
    def encoder_forward(self, images_cur, images_cur_raddino, images_prev, images_prev_raddino, inputs, comprisions):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        # obtain the image_encoder forward
        outputs = self.image_encoder_1(images_cur, images_prev)
        outputs_cur = self.image_encoder_2(images_cur_raddino)
        # print(f"last_hidden_state: {outputs_cur['last_hidden_state'][:, 0, :]}")
        # print(f"pooler_output: {outputs_cur['pooler_output']}")
        outputs_prev = self.image_encoder_2(images_prev_raddino)
        image_cur_embed = torch.cat([outputs_cur['pooler_output'].unsqueeze(dim=1), outputs_cur['last_hidden_state']], dim=1)
        image_prev_embed = torch.cat([outputs_prev['pooler_output'].unsqueeze(dim=1), outputs_prev['last_hidden_state']], dim=1)
        # 分别获取当前影像和历史影像的特征
        # cur_image_embed = outputs['image_feat'][0]
        # prev_image_embed = outputs['image_feat'][1]
        # print('gaile')
        # print(f"cur_image_embed: {cur_image_embed[:, 0, :]}, cur_image_embed dtype: {cur_image_embed.dtype}")

        diff_image_embed = outputs['diff']

        # projection head 将图像特征映射到文本特征相同的特征空间
        # cur_image_embed = self.image_projection_1(cur_image_embed)  # (b, 1371, 768)
        # print(f"cur_image_embed: {cur_image_embed[:, 0, :]}, cur_image_embed dtype: {cur_image_embed.dtype}")
        # prev_image_embed = self.image_projection_1(prev_image_embed)  
        image_cur_embed = self.image_projection_2(image_cur_embed)
        # print(f"image_cur_embed: {image_cur_embed[:, 0, :]}")
        image_prev_embed = self.image_projection_2(image_prev_embed)
        diff_image_embed = self.image_projection_1(diff_image_embed)

        # obtain the text_encoder forward
        text_embed = self.text_encoder(**inputs)
        diff_text_embed = self.text_encoder(**comprisions)
        # print(f"diff_text_embed: {diff_text_embed['last_hidden_state'].shape}")
        text_embed = self.text_projection(text_embed['last_hidden_state'])  # (b, token_num + 1, 768)
        diff_text_embed = self.text_projection(diff_text_embed['last_hidden_state'])
        # print(f"diff_text_embed: {diff_text_embed.shape}")
        # return cur_image_embed, prev_image_embed, diff_image_embed, text_embed, diff_text_embed
        return image_cur_embed, image_prev_embed, diff_image_embed, text_embed, diff_text_embed

        # encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)
        # return encoder_outputs

    def forward(self, images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports, contexts, comprisons, patient_ids, is_aligned):
    # def forward(self, images_cur, images_prev, reports, contexts, comprisons, patient_ids, is_aligned):    
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        device = images_cur.device
        images_prev = images_prev.to(device)
        report_inputs = self.tokenization(reports, device=device)
        comprisions = self.tokenization(comprisons, device=device)

        batch_size = len(reports)
        # cur_image_embed, prev_image_embed, diff_image_embed, text_embed, diff_text_embed = self.encoder_forward(images_cur, images_prev, report_inputs, comprisions)
        image_cur_embed, image_prev_embed, diff_image_embed, text_embed, diff_text_embed = self.encoder_forward(images_cur, images_cur_raddino, images_prev, images_prev_raddino, report_inputs, comprisions)

        # before contrastive learning, it should add vp_pos_embed
        if self.args.is_multiview_fusion:
            # calculate multiview-enhanced/guided contrastive learning among images
            # note that image_embed has not [cls] token, and we treat global average pooling of image_embed
            # its global_feats
            # multiview fusion based on cross-attention
            # 使用全局特征对齐
            # image_embed = self.multiview_fusion_network(cur_image_embed, prev_image_embed, batch_size)
            image_embed = self.multiview_fusion_network(image_cur_embed, image_prev_embed, batch_size)
        else:
            image_embed = image_cur_embed[:batch_size]
            # image_embed = cur_image_embed[:batch_size]
            # print("请调试multiview_fusion_network函数部分！")

        # ====instance-level contrastive loss====
        instance_loss = self.global_alignment_loss(image_embed[:, 0, :], text_embed[:, 0, :], patient_ids)
        # ====diff-level contrastive loss====
        if self.args.using_diff_loss:
        # if self.args['using_diff_loss']:
        # TODO；更改为全局特征再跑一次
            # diff_loss = self.diff_alignment_loss(diff_image_embed, diff_text_embed[:, 1:, :], is_aligned)
            diff_loss = self.diff_alignment_loss(diff_image_embed, diff_text_embed[:, 0, :], is_aligned)
        # ====sentence-level contrastive loss====
        if self.args.using_local_loss:
            sen_text_loss = self.local_text_token_alignment_loss(image_embed[:, 1:, :], text_embed[:, 1:, :])
            if self.args.using_diff_loss:
            # if self.args['using_diff_loss']:
                return {
                    'sen_text_loss': sen_text_loss,
                    'instance_loss': instance_loss,
                    'diff_loss': diff_loss,
                    'loss': instance_loss + sen_text_loss + diff_loss
                }
            else:
                return {
                    'sen_text_loss': sen_text_loss,
                    'instance_loss': instance_loss,
                    'loss': instance_loss + sen_text_loss
                }
        else:
            if self.args.using_diff_loss:
            # if self.args['using_diff_loss']:
                return {
                    'instance_loss': instance_loss,
                    'mpc_loss': diff_loss,
                    'loss': instance_loss + diff_loss
                }
            else:
                return {
                    'instance_loss': instance_loss,
                    'loss': instance_loss
                }

    def training_step(self, batch, batch_idx):
        # print("training_step")
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # image_ids, images, reports, patient_ids, view_positions = batch
        # comprisons描述两张影像之间变化的文本
        image_ids, images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports, contexts, comprisions, patient_ids, is_aligned = batch
        # Inference:
        loss_dict = self(images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports, contexts, comprisions, patient_ids, is_aligned)
        # loss_dict = self(images_cur, images_prev, reports, contexts, comprisions, patient_ids, is_aligned)

        self.log_dict({f'train_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=len(reports),
                      prog_bar=True, sync_dist=True)
        if batch_idx % self.args.print_step == 0 or batch_idx + 1 == self.trainer.num_training_batches:
        # if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.mylog.info(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update loss through mean_metric
        for key, loss in loss_dict.items():
            if f"{key}" in self.train_loss_metric:
                self.train_loss_metric[f"{key}"].update(loss.detach())

        # Update and log scores for each validation metric:
        return loss_dict['loss']
    
    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        image_ids, images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports, contexts, comprisions, patient_ids, is_aligned= batch
        # Inference:
        loss_dict = self(images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports, contexts, comprisions, patient_ids, is_aligned)
        # loss_dict = self(images_cur, images_prev, reports, contexts, comprisions, patient_ids, is_aligned)

        # Logging:
        self.log_dict({f'val_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports),
                      prog_bar=True, sync_dist=True)

        if batch_idx % self.args.print_step == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
        # if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.mylog.info(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        for key, loss in loss_dict.items():
            if f"{key}" in self.val_loss_metric:
                self.val_loss_metric[f"{key}"].update(loss)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        image_ids, images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports, contexts, comprisions, patient_ids, is_aligned = batch
        # Inference:
        loss_dict = self(images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports, contexts, comprisions, patient_ids, is_aligned)
        # loss_dict = self(images_cur, images_prev, reports, contexts, comprisions, patient_ids, is_aligned)

        # Logging:
        self.log_dict({f'test_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports),
                      prog_bar=True, sync_dist=True)
        if batch_idx % self.args.print_step == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
        # if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.mylog.info(f"Epoch {self.current_epoch}, testing step {batch_idx}/{self.trainer.num_test_batches[0]}, "
                            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")
        for key, loss in loss_dict.items():
            if f"{key}" in self.test_loss_metric:
                self.test_loss_metric[f"{key}"].update(loss)

    def on_train_epoch_end(self):
        # print("on_train_epoch_end")
        cur_all_loss = {}
        for key, metric in self.train_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'train_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True,
                      on_step=False, prog_bar=True)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.mylog.info(
            f"Epoch {self.current_epoch}, Training is over, "
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        cur_all_loss = {}
        for key, metric in self.val_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'val_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False, prog_bar=True)

        if cur_all_loss['loss'] < self.val_min_losses["loss"]:
            self.val_min_losses = {**cur_all_loss, "epoch": self.current_epoch}

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        best_loss_item = ', '.join([f"{k} = {v}" for k, v in self.val_min_losses.items()])
        self.mylog.info(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current val loss:"
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
            f"best validation loss: {best_loss_item}\n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """
        cur_all_loss = {}
        for key, metric in self.test_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'test_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False, prog_bar=True)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.mylog.info(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, test is over, current loss:"
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
        )

class Finetune(pl.LightningModule):
    def __init__(
            self,
            args,
            tokenizer: GPT2TokenizerFast,
            logger,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mylog = logger
        # 保留确定性设置，但是只警告，不报错
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.prefetch_factor = None
        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": -1.0,
        }
        self.time_sum = 0

        self.train_loss_metric = torchmetrics.MeanMetric()

        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"], save=False)

        self.val_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args.chexbert_path,
            model_path=args.bert_path,
            mbatch_size=16,
            exp_dir=args.exp_dir_trial,
        )
        self.test_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args.chexbert_path,
            model_path=args.bert_path,
            mbatch_size=16,
            exp_dir=args.exp_dir_trial,
        )
        # Radgraph metrics:
        self.val_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args.radgraph_path,
            mbatch_size=16,
            exp_dir=args.exp_dir_trial,
        )
        self.test_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args.radgraph_path,
            mbatch_size=16,
            exp_dir=args.exp_dir_trial,
        )

        # Report logging:
        self.val_report_logger = ReportLogger(exp_dir=args.exp_dir_trial, split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=args.exp_dir_trial, split='test_reports')

        # Image Diff Encoder1
        # 可以直接从CheXRelFormer_FeatureExtractor提取到特征，只需要写一个接口传入就可以了
        self.image_encoder_1 = CheXRelFormer_FeatureExtractor().to(args.device)
        # 因为图像编码器是我自己定义的，我知道隐藏层维度，直接就写死了
        try:
            image_dim = self.image_encoder_1.config.hidden_size
            self.image_encoder_1.eval()
            for param in self.image_encoder_1.parameters():
                param.requires_grad = False
        except AttributeError:
            # 如果 config 或 hidden_size 不存在，回退到默认值
            image_dim = 768
        
        #  Image Local & Global Encoder2
        self.image_processor = AutoImageProcessor.from_pretrained(args.rad_dino_path)
        self.image_encoder_2 = AutoModel.from_pretrained(args.rad_dino_path)
        # 获取图像编码器的隐藏层为度，在后续的特征投影或者融合中使用
        image_dim_2 = self.image_encoder_2.config.hidden_size
        self.image_encoder_2.eval()
        for param in self.image_encoder_2.parameters():
            param.requires_grad = False

        # Text Encoder
        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # projection head
        self.image_projection_1 = ProjectionHead(image_dim, args.hidden_size * 2, args.hidden_size)
        self.image_projection_2 = ProjectionHead(image_dim_2, args.hidden_size * 2, args.hidden_size)
        self.text_projection = ProjectionHead(text_dim, args.hidden_size * 2, args.hidden_size)

        # layer_norm
        # self.ln_1 = nn.LayerNorm(image_dim)
        # self.ln_2 = nn.LayerNorm(args.hidden_size)
        self.ln_1 = nn.LayerNorm(args.hidden_size)

        # temp_pos_embed for temporal information (0 for ori_image_fea, 1 for temporal_fea)
        self.type_pos_embed = nn.Parameter(torch.rand(2, 1, args.hidden_size), requires_grad=True)

        # # fusion module
        self.fusion_multiview = Transformer(args.hidden_size, args.multiview_fusion_num_layers,
                                            heads=args.num_heads,
                                            dim_head=args.hidden_size // 4,
                                            mlp_dim=args.hidden_size)

        
        self.text_decoder = self.build_text_decoder()

        # cross-attention fusion network
        # 让影像特征 selectively 参考 prompt 内容
        fusion_multimodal_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.bert_path,
            vocab_size=len(self.tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.cross_modal_fusion_num_layers,
            num_attention_heads=args.num_heads,
            max_position_embeddings=512,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        self.fusion_multimodal = nn.ModuleList(
            [BertCrossLayer(fusion_multimodal_config) for _ in range(args.cross_modal_fusion_num_layers)])

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args.cxr_bert_path, trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args.text_encoder_num_layers
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args.cxr_bert_path,
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    def build_text_decoder(self):
        config = transformers.GPT2Config.from_pretrained(self.args.distilgpt2_path)
        config.add_cross_attention = True
        config.is_decoder = True
        config.vocab_size = len(self.tokenizer)
        if self.args.cvt2distilgpt2_path is None:
            decoder = transformers.GPT2LMHeadModel.from_pretrained(
                self.args.distilgpt2_path,
                config=config,
                ignore_mismatched_sizes=True
            )
            # Resize GPT2 embedding to include padding and beginning of sentence token:
            decoder.resize_token_embeddings(len(self.tokenizer))
        else:
            print(f"Loading DistilGPT2 from CVT checkpoint...{self.args.cvt2distilgpt2_path}")
            decoder = transformers.GPT2LMHeadModel(config=config)
            # Resize GPT2 embedding to include padding and beginning of sentence token:
            decoder.resize_token_embeddings(len(self.tokenizer))

            checkpoint = torch.load(self.args.cvt2distilgpt2_path)['state_dict']
            checkpoint = {k.split('decoder.encoder_decoder.decoder.')[-1]: v for k, v in checkpoint.items() if
                          'decoder' in k}
            curr_state_dict = decoder.state_dict()
            valid_state_dict = {k: v for k, v in checkpoint.items() if
                                k in curr_state_dict and v.shape == curr_state_dict[k].shape}
            curr_state_dict.update(valid_state_dict)
            decoder.load_state_dict(curr_state_dict)

        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def forward(self, *args, **kwargs):
                pass

            def get_output_embeddings(cls):
                return None

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

        return Decoder()

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = ReportDataset(self.args, self.tokenizer, 'train')
            self.val_set = ReportDataset(self.args, self.tokenizer, 'val')
            # print(
            #     "No. of training & validation examples: {} & {}.".format(
            #         self.train_set.__len__(), self.val_set.__len__()
            #     )
            # )
            # self.mylog.info("No. of training & validation examples: {} & {}.".format(
            #     self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:  # fit
            self.test_set = ReportDataset(self.args, self.tokenizer, 'test')
            # print("No. of test examples: {}.".format(self.test_set.__len__()))
            # self.mylog.info("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        collate_fn = DataLoaders.finetune_collate_fn
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        collate_fn = DataLoaders.finetune_collate_fn
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        collate_fn = DataLoaders.finetune_collate_fn
        return DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        if self.args.task == 'pretrain':
            optimiser = torch.optim.AdamW(self.parameters(), lr=self.args.pt_lr)
            lr_scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': 'val_loss',
                    'frequency': 1   # the frequency of check
                }
            }
        else:
            pretrain_main_params, finetune_main_params = [], []
            if self.args.load is not None:
                print(f"Loading pre-trained checkpoint from {self.args.load}...")
                checkpoint = torch.load(self.args.load)['state_dict']
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in checkpoint:
                        pretrain_main_params.append(param)
                    else:
                        finetune_main_params.append(param)
            else:  # all parameters are finetuning
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    finetune_main_params.append(param)

            optimiser = torch.optim.AdamW(
                [{'params': pretrain_main_params, 'lr': self.args.pt_lr},
                 {'params': finetune_main_params, 'lr': self.args.ft_lr}])

            lr_scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=5)
            return {
                "optimizer": optimiser,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.args.monitor_metric,
                    'frequency': 1   # the frequency of check
                }
            }

    def tokenization(self, text, pair_text=None, device=None):
        if pair_text is None:
            inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=True,
                                    max_length=self.args.max_length + 1,  # As we remove a token below.
                                    truncation=True)
        else:
            inputs = self.tokenizer(text, pair_text, padding=True, return_token_type_ids=True,
                                    return_tensors='pt', max_length=200, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        return inputs
    
    # 注意调试这里
    def obtain_decoder_input_ids(self, inputs):
        decoder_input_ids = inputs['input_ids']
        decoder_attention_mask = inputs['attention_mask'][:, :-1]  # string + [eos]
        label_ids = decoder_input_ids[:, 1:].detach().clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_input_ids[decoder_input_ids == self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id
        return decoder_input_ids, decoder_attention_mask, label_ids

    def obtain_reference_reports(self, text):
        inputs = self.tokenizer(text, padding=True, max_length=self.args.max_length,
                                truncation=True, return_tensors='pt')
        ref_reports = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        return ref_reports

    def multiview_fusion_network(self, cur_image_embed, prev_image_embed , batch_size):
        # obtain labels indicate corresponding multiview images
        # 只融合历史影像
        new_image_embed = []
        for i in range(batch_size):
            # include multiview images
            image_embed = self.fusion_multiview(cur_image_embed[i], prev_image_embed[i],
                                                prev_image_embed[i])

            new_image_embed.append(image_embed)
        new_image_embed = torch.stack(new_image_embed, dim=0)
        return new_image_embed

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        # obtain multi-positive target
        # 对 patient_ids 进行裁剪，使其长度与 global_image_embed 的第 0 维（即样本数量）保持一致。
        # print('global_image_embed shape: ', global_image_embed.shape)
        patient_ids = patient_ids[:global_image_embed.shape[0]]
        # print('patient_ids: ', patient_ids)
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels).float().to(global_image_embed.device)
        labels = labels / labels.sum(1, keepdim=True)
        del patient_ids

        # normalize
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        global_text_embed = F.normalize(global_text_embed, dim=-1, p=2)

        # calculate the InfoNCE loss
        instance_sim = global_image_embed @ global_text_embed.t()
        instance_sim_1 = global_text_embed @ global_image_embed.t()
        # temp控制相似度分布的尖锐程度
        loss_instance_1 = F.cross_entropy(instance_sim / self.args.temp, labels)
        loss_instance_2 = F.cross_entropy(instance_sim_1 / self.args.temp, labels)
        global_instance_loss = (loss_instance_1 + loss_instance_2) / 2.0
        return global_instance_loss

    def local_text_token_alignment_loss(self, local_image_embed, local_text_embed):
        # cross-modal alignment between image patches and sentence embed in reports

        t_att_sim = local_text_embed @ local_image_embed.permute(0, 2, 1).contiguous()
        t_att_sco = F.softmax(t_att_sim / math.sqrt(local_image_embed.shape[2]), dim=-1)
        t_att_output = torch.bmm(t_att_sco, local_image_embed)

        device = local_image_embed.device
        # normalize
        t_att_output = F.normalize(t_att_output, dim=-1, p=2)
        local_text_embed = F.normalize(local_text_embed, dim=-1, p=2)
        # calculate the loss
        word_sim = torch.bmm(local_text_embed, t_att_output.permute(0, 2, 1).contiguous()) / self.args['region_temp']
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        loss_word = (loss_word_2 + loss_word_1) / 2.0
        return loss_word

    def diff_alignment_loss(self, diff_image_embed, diff_text_embed):
        # cross-modal alignment between diff_image patches and sentence embed in reports

        t_att_sim = diff_text_embed @ diff_image_embed.permute(0, 2, 1).contiguous()
        t_att_sco = F.softmax(t_att_sim / math.sqrt(diff_image_embed.shape[2]), dim=-1)
        t_att_output = torch.bmm(t_att_sco, diff_image_embed)

        device = diff_image_embed.device
        # normalize
        t_att_output = F.normalize(t_att_output, dim=-1, p=2)
        diff_text_embed = F.normalize(diff_text_embed, dim=-1, p=2)
        # calculate the loss
        word_sim = torch.bmm(diff_text_embed, t_att_output.permute(0, 2, 1).contiguous()) / self.args.region_temp
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        diff_loss = (loss_word_2 + loss_word_1) / 2.0

        return diff_loss
    # TODO:fixed
    def text_encoder_forward(self, inputs):
        # obtain the text_encoder forward
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])  # (b, token_num, 768)
        return text_embed
    def image_encoder_forward(self, images_cur, images_cur_raddino, images_prev, images_prev_raddino):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """

        # obtain the image_encoder forward
        outputs = self.image_encoder_1(images_cur, images_prev)
        outputs_cur = self.image_encoder_2(images_cur_raddino)
        # print(f"outputs_cur: {outputs_cur['pooler_output']}")
        outputs_prev = self.image_encoder_2(images_prev_raddino)
        image_cur_embed = torch.cat([outputs_cur['pooler_output'].unsqueeze(dim=1), outputs_cur['last_hidden_state']], dim=1)
        image_prev_embed = torch.cat([outputs_prev['pooler_output'].unsqueeze(dim=1), outputs_prev['last_hidden_state']], dim=1)
        # 分别获取当前影像和历史影像的特征
        # cur_image_embed = outputs['image_feat'][0]
        # prev_image_embed = outputs['image_feat'][1]
        # print('gaile')
        # print(f"cur_image_embed: {cur_image_embed[:, 0, :]}, cur_image_embed dtype: {cur_image_embed.dtype}")

        diff_image_embed = outputs['diff']

        # projection head 将图像特征映射到文本特征相同的特征空间
        # cur_image_embed = self.image_projection_1(cur_image_embed)  # (b, 1371, 768)
        # print(f"cur_image_embed: {cur_image_embed[:, 0, :]}, cur_image_embed dtype: {cur_image_embed.dtype}")
        # prev_image_embed = self.image_projection_1(prev_image_embed)  
        image_cur_embed = self.image_projection_2(image_cur_embed)
        # print(f"image_cur_embed: {image_cur_embed[:, 0, :]}")
        image_prev_embed = self.image_projection_2(image_prev_embed)
        diff_image_embed = self.image_projection_1(diff_image_embed)

        return image_cur_embed, image_prev_embed, diff_image_embed
    
    def forward(self, images_cur, images_cur_raddino, images_prev, images_prev_raddino, patient_ids, indications, contexts, reports=None, mode='train'):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        # basic config
        device = images_cur.device
        images_prev = images_prev.to(device)
        batch_size = len(contexts)

        # obtain the prompt_embed (including indications and prior_reports)
        # token embedding + position embedding + segment embedding
        prompt_embed = None
        if self.args.is_indication:
            if self.args.is_prior_report:
                prompt_inputs = self.tokenization(indications, pair_text=contexts, device=device)
                prompt_embed = self.text_encoder_forward(prompt_inputs)
            else:
                prompt_inputs = self.tokenization(indications, pair_text=None, device=device)
                prompt_embed = self.text_encoder_forward(prompt_inputs)
        else:
            if self.args.is_prior_report:
                prompt_inputs = self.tokenization(contexts, pair_text=None, device=device)
                prompt_embed = self.text_encoder_forward(prompt_inputs)
        

        image_cur_embed, image_prev_embed, diff_image_embed = self.image_encoder_forward(images_cur, images_cur_raddino, images_prev, images_prev_raddino)
        # print("cur_image_embed shape: ", cur_image_embed.shape)
        # print("prev_image_embed shape: ", prev_image_embed.shape)
        # 可能需要小小的修改一下
        ori_image_embed = image_cur_embed[:batch_size] + torch.cat([self.type_pos_embed[0].unsqueeze(0)] * batch_size,
                                                               dim=0)

        if self.args.is_multiview_fusion:
            # calculate multiview-enhanced/guided contrastive learning among images
            # multiview fusion based on cross-attention
            image_embed = self.multiview_fusion_network(image_cur_embed, image_prev_embed, batch_size)
        else:
            image_embed = image_cur_embed[:batch_size]
        #     image_embed = self.multiview_fusion_network(cur_image_embed, prev_image_embed, batch_size)
        # else:
        #     # add temporal positional embedding
        #     # image_embed = image_embed[:batch_size]
        #      print("请调试multiview_fusion_network函数部分！")

        # cat ori_image_embed, tempor_image_embed
        image_embed = image_embed + torch.cat([self.type_pos_embed[1].unsqueeze(0)] * batch_size, dim=0)
        diff_image_embed = diff_image_embed + torch.cat([self.type_pos_embed[1].unsqueeze(0)] * batch_size, dim=0)
        image_embed = torch.cat([ori_image_embed, diff_image_embed], dim=1)
        image_embed = self.ln_1(image_embed)

        if prompt_embed is not None:
            # integrate prompt information using cross-attention
            encoder_attention_mask = torch.ones(image_embed.size()[:2], dtype=torch.long).to(device)
            extended_image_masks = get_extended_attention_mask(encoder_attention_mask, encoder_attention_mask.size())
            extended_text_masks = get_extended_attention_mask(prompt_inputs['attention_mask'], prompt_embed.size())

            x, y = image_embed.clone(), prompt_embed
            for layer_idx, image_layer in enumerate(self.fusion_multimodal):
                x1 = image_layer(x, y, attention_mask=extended_image_masks,
                                 encoder_attention_mask=extended_text_masks, output_attentions=True)
                x = x1[0]
            encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=x)
        else:
            encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_embed)
        if mode == 'train':
            report_inputs = self.tokenization(reports, device=device)
            decoder_input_ids, decoder_attention_mask, labels_ids = self.obtain_decoder_input_ids(report_inputs)
            # Teacher forcing: labels are given as input
            outputs = self.text_decoder.encoder_decoder(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                labels=labels_ids
            )
            return outputs['loss']
        else:
            outputs = self.generate(encoder_outputs)
            generated_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return generated_reports

    def generate(self, encoder_outputs):
        """
        Autoregressive generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        outputs = self.text_decoder.encoder_decoder.generate(
            # special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.args.max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.args.num_beams,
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )

        return outputs['sequences']

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        images_id, images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports_base, indications, contexts, patient_ids = batch
        # Inference:
        loss = self(images_cur, images_cur_raddino, images_prev, images_prev_raddino, patient_ids, indications, contexts, reports=reports_base, mode='train')

        self.log_dict({'lm_loss': loss}, on_step=True, on_epoch=True, batch_size=len(reports_base),
                      prog_bar=True, sync_dist=True)
        self.train_loss_metric.update(loss)
        if batch_idx % self.args.print_step == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            self.mylog.info(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{loss.detach().item()}, lr: {self.optimizers().param_groups[0]['lr']},"
                f"{self.optimizers().param_groups[1]['lr']}")
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        image_ids, images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports_base, indications, contexts, patient_ids = batch
        # Inference:
        generated_reports = self(images_cur, images_cur_raddino, images_prev, images_prev_raddino, patient_ids, indications, contexts,
                                 reports=None, mode='sample')
        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(reports_base)  # remove special tokens

        if batch_idx % self.args.print_step == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.mylog.info(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        # # Log reports:
        self.val_report_logger.update(generated_reports, dicom_ids=image_ids, labels=reference_reports)

        # # Evaluate:
        # self.val_chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=image_ids)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        image_ids, images_cur, images_cur_raddino, images_prev, images_prev_raddino, reports_base, indications, contexts, patient_ids = batch
        # Inference:
        start = time.time()
        generated_reports = self(images_cur, images_cur_raddino, images_prev, images_prev_raddino, patient_ids, indications, contexts,
                                 reports=None, mode='sample')
        end = time.time()
        self.time_sum += end - start
        reference_reports = self.obtain_reference_reports(reports_base)  # remove special tokens

        if batch_idx % self.args.print_step == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.mylog.info(
                f"Testing step {batch_idx}/{self.trainer.num_test_batches[0]}")

        # Log reports:
        self.test_report_logger.update(generated_reports, dicom_ids=image_ids, labels=reference_reports)
        #
        # # Evaluate:
        # self.test_chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_radgraph_metrics.update(generated_reports, reference_reports, ids=image_ids)

    def on_train_epoch_end(self):
        epoch_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        self.mylog.info(
            f"Epoch {self.current_epoch}, Training is over, "
            f"epoch lm_loss = {epoch_loss}, lr: {self.optimizers().param_groups[0]['lr']}, "
            f"{self.optimizers().param_groups[1]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()
        #
        scores = {}
        # F1-radgraph
        output = self.val_radgraph_metrics.compute()
        scores.update(output)
        self.val_radgraph_metrics.reset()

        # chexbert
        output = self.val_f1chexbert_metrics.compute()
        scores.update(output)
        self.val_f1chexbert_metrics.reset()

        # output = self.val_chexbert_metrics.compute()
        # scores.update(output)
        # self.val_chexbert_metrics.reset()
        #
        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

        if scores[self.args.monitor_metric] > self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor_metric': scores[self.args.monitor_metric]
            }

        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.mylog.info(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current metrics:\n"
            f"best validation epoch: {self.val_best_scores['best_epoch']}, "
            f"best val_metrics: {self.args.monitor_metric} = {self.val_best_scores['best_monitor_metric']}\n"
            f"{metrics_item} \n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """
        print(f"all time is {self.time_sum}, the average time of each image is {self.time_sum / len(self.test_set)}")

        # Save reports:
        self.test_report_logger.log(1)
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}
        output = self.test_radgraph_metrics.compute()
        scores.update(output)
        self.test_radgraph_metrics.reset()

        # output = self.test_chexbert_metrics.compute()
        # scores.update(output)
        # self.test_chexbert_metrics.reset()

        output = self.test_f1chexbert_metrics.compute() 
        scores.update(output)
        self.test_f1chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        print('\n')
        print(scores)

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.mylog.info(
            "###############################################################\n"
            f"test is over, current metrics:"
            f"{metrics_item} \n"
        )