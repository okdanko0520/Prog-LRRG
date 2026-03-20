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

        # Image Encoder:
        # self.image_processor = AutoImageProcessor.from_pretrained(args.rad_dino_path)
        # self.image_encoder = AutoModel.from_pretrained(args.rad_dino_path)
        # self.image_encoder.eval()
        # image_dim = self.image_encoder.config.hidden_size
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False
        self.image_encoder = CheXRelFormer_FeatureExtractor().to(args.device)
        for param in self.image_encoder.parameters():
            print('param device: ', param.device)
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
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # projection head
        self.image_projection = ProjectionHead(image_dim, args.hidden_size * 2, args.hidden_size)
        self.text_projection = ProjectionHead(text_dim, args.hidden_size * 2, args.hidden_size)

        # layer_norm
        self.ln_1 = nn.LayerNorm(image_dim)
        self.ln_2 = nn.LayerNorm(args.hidden_size)
        self.ln_3 = nn.LayerNorm(args.hidden_size)

        # temp_pos_embed for temporal information (0 for ori_image_fea, 1 for temporal_fea)
        self.type_pos_embed = nn.Parameter(torch.rand(3, 1, args['hidden_size']), requires_grad=True)

        # # fusion module
        self.fusion_multiview = Transformer(args.hidden_size, args.multiview_fusion_num_layers,
                                            heads=args.num_heads,
                                            dim_head=args.hidden_size // 4,
                                            mlp_dim=args.hidden_size)

        # TODO:注意修改这里
        # # Decoder:
        # # ckpt_name = 'distilbert/distilgpt2'
        self.text_decoder = self.build_text_decoder()

        # cross-attention fusion network
        fusion_multimodal_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args['bert_path'],
            vocab_size=len(self.tokenizer),
            hidden_size=args["hidden_size"],
            num_hidden_layers=args["cross_modal_fusion_num_layers"],
            num_attention_heads=args["num_heads"],
            max_position_embeddings=512,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        self.fusion_multimodal = nn.ModuleList(
            [BertCrossLayer(fusion_multimodal_config) for _ in range(args['cross_modal_fusion_num_layers'])])

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
            self.train_set = ReportDataset(self.args, 'train', self.tokenizer)
            self.val_set = ReportDataset(self.args, 'val', self.tokenizer)
            # print(
            #     "No. of training & validation examples: {} & {}.".format(
            #         self.train_set.__len__(), self.val_set.__len__()
            #     )
            # )
            # self.mylog.info("No. of training & validation examples: {} & {}.".format(
            #     self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:  # fit
            self.test_set = ReportDataset(self.args, 'test', self.tokenizer)
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
            num_workers=self.args['num_workers'],
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
        if self.args['task'] == 'pretrain':
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
        inputs = self.tokenizer(text, padding=True, max_length=self.args['max_length'],
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
        print('global_image_embed shape: ', global_image_embed.shape)
        patient_ids = patient_ids[:global_image_embed.shape[0]]
        print('patient_ids: ', patient_ids)
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
    
    def text_encoder_forward(self, inputs):
        # obtain the text_encoder forward
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])  # (b, token_num, 768)
        return text_embed

    def image_encoder_forward(self, images_cur, images_prev):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        # obtain the image_encoder forward
        outputs = self.image_encoder(images_cur, images_prev)
        # 分别获取当前影像和历史影像的特征
        cur_image_embed = outputs['image_feat'][0]
        prev_image_embed = outputs['image_feat'][1]
        diff_image_embed = outputs['diff']
        # 不用拼接
        # image_embed = torch.cat([outputs['pooler_output'].unsqueeze(dim=1), outputs['last_hidden_state']], dim=1)
        # image_embed = self.image_encoder(images)['last_hidden_state']  # (b, 384, 576)
        # add view_position embedding (positional embedding)
        # valid_view_positions = [vp.split('_')[0] for vp in view_positions]
        # image_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in valid_view_positions]
        # add pos_embed & add & norm
        # image_embed = torch.cat(image_pos_embed, dim=0) + image_embed
        # image_embed = self.ln_1(image_embed)
        # projection head 将图像特征映射到文本特征相同的特征空间
        cur_image_embed = self.image_projection(cur_image_embed)  # (b, 1371, 768)
        prev_image_embed = self.image_projection(prev_image_embed)  
        diff_image_embed = self.image_projection(diff_image_embed)

        return cur_image_embed, prev_image_embed, diff_image_embed
    
    def forward(self, images_cur, images_prev, patient_ids, contexts, reports=None, mode='train'):
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
        if self.args.is_prior_report:
            prompt_inputs = self.tokenization(contexts, pair_text=None, device=device)
            prompt_embed = self.text_encoder_forward(prompt_inputs)

        cur_image_embed, prev_image_embed, diff_image_embed = self.image_encoder_forward(images_cur, images_prev)
        # 可能需要小小的修改一下
        ori_image_embed = cur_image_embed[:batch_size] + torch.cat([self.type_pos_embed[0].unsqueeze(0)] * batch_size,
                                                               dim=0)

        if self.args.is_multiview_fusion:
            # calculate multiview-enhanced/guided contrastive learning among images
            # multiview fusion based on cross-attention
            image_embed = self.multiview_fusion_network(cur_image_embed, prev_image_embed, batch_size)
        else:
            # add temporal positional embedding
            # image_embed = image_embed[:batch_size]
             print("请调试multiview_fusion_network函数部分！")

        # cat ori_image_embed, tempor_image_embed
        image_embed = image_embed + torch.cat([self.type_pos_embed[1].unsqueeze(0)] * batch_size, dim=0)
        diff_image_embed = diff_image_embed + torch.cat([self.type_pos_embed[2].unsqueeze(0)] * batch_size, dim=0)
        image_embed = torch.cat([ori_image_embed, image_embed, diff_image_embed], dim=1)
        image_embed = self.ln_3(image_embed)

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
        image_ids, images_cur, images_prev, reports_base, contexts, patient_ids = batch
        # Inference:
        loss = self(images_cur, images_prev, patient_ids, contexts, reports=reports_base, mode='train')

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
        image_ids, images_cur, images_prev, reports_base, contexts, patient_ids = batch
        # Inference:
        generated_reports = self(images_cur, images_prev, patient_ids, contexts,
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
        image_ids, images_cur, images_prev, reports_base, contexts, patient_ids = batch
        # Inference:
        start = time.time()
        generated_reports = self(images_cur, images_prev, patient_ids, contexts,
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