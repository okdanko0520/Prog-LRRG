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
from model.bert_model import Transformer, BertCrossLayer

from ext.resnet import ProjectionHead
from modules.CheXRelFormer import CheXRelFormer_FeatureExtractor


class Pretrain(nn.Module):
    def __init__(self, args, tokenizer, logger, **kwargs):
        super(Pretrain, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.myloag = logger
        # 每个工作线程会预先获得5个数据
        self.prefetch_factor = 5

        # Image Encoder
        # TODO:修改视觉提取模块，！！！添加差异特征的获取部分
        # 可以直接从CheXRelFormer_FeatureExtractor提取到特征，只需要写一个接口传入就可以了
        # self.image_processor = AutoImageProcessor.from_pretrained(args.vit_path)
        self.image_encoder = CheXRelFormer_FeatureExtractor().to(args.device)  #加了to(device)
        # 获取图像编码器的隐藏层维度，在后续的特征投影或者融合中使用
        # 因为图像编码器是我自己定义的，我知道隐藏层维度，直接就写死了
        # self.image_dim = self.image_encoder.config.hidden_size
        try:
            image_dim = self.image_encoder.config.hidden_size
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        except AttributeError:
            # 如果 config 或 hidden_size 不存在，回退到默认值
            image_dim = 768

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
        self.image_projection = ProjectionHead(image_dim, args.hidden_size * 2, args.hidden_size)
        # 文本投影，于将文本编码器生成的高维特征（text_dim）投影到与图像特征相同的特征空间。
        self.text_projection = ProjectionHead(text_dim, args.hidden_size* 2, args.hidden_size)

        # layer_norm--归一化
        self.ln_1 = nn.LayerNorm(image_dim)
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
    
    def multiview_fusion_network(self, image_embed, patient_ids, batch_size):
        # obtain labels indicate corresponding multiview images
        # 只融合历史影像
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels)
        labels.fill_diagonal_(0)

        new_image_embed = []
        for i in range(batch_size):
            if labels[i].sum() == 0:
                new_image_embed.append(image_embed[i])
                continue
            multiview_image_embed = torch.cat([image_embed[j] for j, tag in enumerate(labels[i]) if tag == 1], dim=0)
            # include multiview images
            cur_image_embed = self.fusion_multiview(image_embed[i], multiview_image_embed,
                                                    multiview_image_embed)

            new_image_embed.append(cur_image_embed)
        new_image_embed = torch.stack(new_image_embed, dim=0)
        return new_image_embed

    def diff_alignment_loss(self, diff_image_embed, diff_text_embed):
        # normalize
        diff_image_embed = F.normalize(diff_image_embed, dim=-1, p=2)
        diff_text_embed = F.normalize(diff_text_embed, dim=-1, p=2)
        # similarity
        instance_sim = diff_image_embed @ diff_text_embed.t()
        instance_sim_1 = diff_text_embed @ diff_image_embed.t()
        # labels
        batch_size = diff_image_embed.shape[0]
        labels = torch.arange(batch_size).to(diff_image_embed.device)

        # InfoNCE loss
        loss_1 = F.cross_entropy(instance_sim / self.args.diff_temp, labels)
        loss_2 = F.cross_entropy(instance_sim_1 / self.args.diff_temp, labels)
        diff_loss = (loss_1 + loss_2) / 2.0
        return diff_loss

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        # obtain multi-positive target
        # 对 patient_ids 进行裁剪，使其长度与 global_image_embed 的第 0 维（即样本数量）保持一致。
        patient_ids = patient_ids[:global_image_embed.shape[0]]
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
        word_sim = torch.bmm(local_text_embed, t_att_output.permute(0, 2, 1).contiguous()) / self.args.region_temp
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        loss_word = (loss_word_2 + loss_word_1) / 2.0
        return loss_word
    
    def encoder_forward(self, images, inputs, view_positions):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        # obtain the image_encoder forward
        outputs = self.image_encoder(images)
        image_embed = torch.cat([outputs['pooler_output'].unsqueeze(dim=1), outputs['last_hidden_state']], dim=1)
        # image_embed = self.image_encoder(images)['last_hidden_state']  # (b, 384, 576)
        # add view_position embedding (positional embedding)
        valid_view_positions = [vp.split('_')[0] for vp in view_positions]
        image_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in valid_view_positions]
        # add pos_embed & add & norm
        image_embed = torch.cat(image_pos_embed, dim=0) + image_embed
        image_embed = self.ln_1(image_embed)
        # projection head
        image_embed = self.image_projection(image_embed)  # (b, 1371, 768)

        # obtain the text_encoder forward
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])  # (b, token_num + 1, 768)

        return image_embed, text_embed

        # encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)
        # return encoder_outputs

    def forward(self, images, reports, patient_ids, view_positions):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        device = images.device
        report_inputs = self.tokenization(reports, device=device)
        batch_size = len(reports)
        image_embed, text_embed = self.encoder_forward(images, report_inputs, view_positions)

        # mul-positive contrastive learning
        mul_pos_loss = torch.tensor([0.0])
        if self.args['using_mpc_loss']:
            mul_pos_loss = self.multiple_positive_contrastive_learning(image_embed[:, 0, :],
                                                                       patient_ids, view_positions)

        # add temporal positional embedding
        temporal_pos_embed = []
        for vp in view_positions:
            if 'prior' not in vp:
                temporal_pos_embed.append(self.temp_pos_embed[0].unsqueeze(0))
            else:
                if 'latest' in vp:
                    temporal_pos_embed.append(self.temp_pos_embed[1].unsqueeze(0))
                else:  # second
                    temporal_pos_embed.append(self.temp_pos_embed[2].unsqueeze(0))
        image_embed = image_embed + torch.cat(temporal_pos_embed, dim=0)
        image_embed = self.ln_2(image_embed)

        # before contrastive learning, it should add vp_pos_embed
        if self.args['is_multiview_learning']:
            # calculate multiview-enhanced/guided contrastive learning among images
            # note that image_embed has not [cls] token, and we treat global average pooling of image_embed
            # its global_feats
            # multiview fusion based on cross-attention
            image_embed = self.multiview_fusion_network(image_embed, patient_ids, batch_size, view_positions)
        else:
            image_embed = image_embed[:batch_size]

        # ====instance-level contrastive loss====
        instance_loss = self.global_alignment_loss(image_embed[:, 0, :], text_embed[:, 0, :], patient_ids)

        # ====sentence-level contrastive loss====
        if self.args['using_local_loss']:
            sen_text_loss = self.local_text_token_alignment_loss(image_embed[:, 1:, :], text_embed[:, 1:, :])
            if self.args['using_mpc_loss']:
                return {
                    'sen_text_loss': sen_text_loss,
                    'instance_loss': instance_loss,
                    'mpc_loss': mul_pos_loss,
                    'loss': instance_loss + sen_text_loss + mul_pos_loss
                }
            else:
                return {
                    'sen_text_loss': sen_text_loss,
                    'instance_loss': instance_loss,
                    'loss': instance_loss + sen_text_loss
                }
        else:
            if self.args['using_mpc_loss']:
                return {
                    'instance_loss': instance_loss,
                    'mpc_loss': mul_pos_loss,
                    'loss': instance_loss + mul_pos_loss
                }
            else:
                return {
                    'instance_loss': instance_loss,
                    'loss': instance_loss
                }

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_mimic_cxr(self, images, context,images2,targets=None,mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        att_feats2, fc_feats2 = self.visual_extractor(images2)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, context,fc_feats2, att_feats2,mode='forward')
        elif mode == 'sample':
            #print(context.shape)
            output, _ = self.encoder_decoder(fc_feats, att_feats,targets, context,fc_feats2, att_feats2,mode='sample')
        else:
            raise ValueError
        return output

