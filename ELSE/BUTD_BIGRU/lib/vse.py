"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import ContrastiveLoss
from lib.loss import UTO

import copy
import logging

logger = logging.getLogger(__name__)

class VSEModel(nn.Module):
    """
        The standard VSE model
    """
    def __init__(self, opt):
        super().__init__()
        # Build Models
        self.grad_clip = opt.grad_clip   # 2.0
        self.img_enc = get_image_encoder(opt.img_dim, opt.embed_size,                     # EncoderImageAggr
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt, use_bi_gru=True, no_txtnorm=opt.no_txtnorm)  # EncoderTex

        if opt.use_moco:
            self.K = opt.moco_M
            self.m = opt.moco_r
            self.v_encoder_k = copy.deepcopy(self.img_enc)
            self.t_encoder_k = copy.deepcopy(self.txt_enc)
            for param in self.v_encoder_k.parameters():
                param.requires_grad = False
            for param in self.t_encoder_k.parameters():
                param.requires_grad = False
            # create the queue
            self.register_buffer("t_queue", torch.rand(opt.embed_size, self.K))  # opt.embed_size:1024; self.K:2048
            self.t_queue = F.normalize(self.t_queue, dim=0)
            self.register_buffer("v_queue", torch.rand(opt.embed_size, self.K))
            self.v_queue = F.normalize(self.v_queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            if opt.use_moco:
                self.v_encoder_k.cuda()
                self.t_encoder_k.cuda()
                self.t_queue = self.t_queue.cuda()
                self.v_queue = self.v_queue.cuda()
                self.queue_ptr = self.queue_ptr.cuda()
            cudnn.benchmark = True
        # MoCo contrastive Loss
        self.hal_loss = UTO(opt=opt)   # opt

        # self_expand_Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params
        self.opt = opt

        self.optimizer = torch.optim.AdamW(self.params, lr=opt.learning_rate)

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.img_enc.parameters(), self.v_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.txt_enc.parameters(), self.t_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_keys, t_keys):
        batch_size = v_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.v_queue[:, ptr : ptr + batch_size] = v_keys.T
        self.t_queue[:, ptr : ptr + batch_size] = t_keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, image_lengths=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_lengths = image_lengths.cuda()
        img_emb = self.img_enc(images, image_lengths)

        lengths = torch.Tensor(lengths).cuda()
        cap_emb = self.txt_enc(captions, lengths)

        if self.opt.use_moco:
            N = images.shape[0]
            with torch.no_grad():
                self._momentum_update_key_encoder()
                v_embed_k = self.v_encoder_k(images, image_lengths)
                t_embed_k = self.t_encoder_k(captions, lengths)

            # loss_moco = self.hal_loss.moco_forward(img_emb, t_embed_k, cap_emb, v_embed_k, self.v_queue, self.t_queue)

            # self._dequeue_and_enqueue(v_embed_k, t_embed_k)

        return img_emb, cap_emb, v_embed_k, t_embed_k


    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, image_lengths=None, is_train = False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_lengths = image_lengths.cuda()
        img_emb = self.img_enc(images, image_lengths)   # (256,1024)

        lengths = torch.Tensor(lengths).cuda()
        cap_emb = self.txt_enc(captions, lengths)

        if is_train and self.opt.use_moco:
            N = images.shape[0]   # N:128
            with torch.no_grad():
                self._momentum_update_key_encoder()
                v_embed_k = self.v_encoder_k(images, image_lengths)
                t_embed_k = self.t_encoder_k(captions, lengths)

            loss_moco = self.hal_loss.moco_forward(img_emb, t_embed_k, cap_emb, v_embed_k, self.v_queue, self.t_queue)
            self._dequeue_and_enqueue(v_embed_k, t_embed_k)
            return img_emb, cap_emb, loss_moco

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        cost_im, cost_s= self.criterion(img_emb, cap_emb)
        self.logger.update('Loss_im', cost_im.item(), cap_emb.size(0))
        self.logger.update('Loss_s', cost_s.item(), cap_emb.size(0))
        loss = cost_im+cost_s
        self.logger.update('Le', loss.item(), cap_emb.size(0))
        return loss


    def train_emb(self, images, captions, caption_lengths, image_lengths=None, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        images_all = images
        image_lens = image_lengths.reshape(-1)
        captions_all = captions.reshape(captions.size(0)*captions.size(1) ,captions.size(2))
        caption_lens = caption_lengths.reshape(-1)

        # compute the embeddings
        if self.opt.use_moco:
            img_emb, cap_emb, loss_moco = self.forward_emb(images_all, captions_all, caption_lens,
                                                           image_lengths=image_lens, is_train=True)
            self.logger.update('Le_moco', loss_moco.data.item(), img_emb.size(0))
            loss_train = self.forward_loss(img_emb, cap_emb)
            loss = loss_train/1000000 + loss_moco
            logit = 60.0
            logit_scale = torch.tensor(logit)
            pid = torch.eye(256)
            # sdm_loss = compute_sdm(img_emb, cap_emb, pid, logit_scale)
            self.logger.update('Loss', loss.data.item(), img_emb.size(0))
        else:
            img_emb, cap_emb, = self.forward_emb(images_all, captions_all, caption_lens, image_lengths=image_lens)
            loss = self.forward_loss(img_emb, cap_emb)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        if warmup_alpha is not None:
            loss = loss * warmup_alpha  # linear lr warmup

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()