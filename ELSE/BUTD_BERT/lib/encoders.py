"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from transformers import BertModel

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.mlp import MLP

import logging

logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(data_name, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.griddrop1 = nn.Dropout(0.2)
        self.griddrop2 = nn.Dropout(0.2)
        self.init_weights()
        self.v_global_w = VisualSA(embed_size, 0.4, 36)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image):
        # Extract image feature vectors.
        features = self.fc(image)

        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for embedding transformation
            features = self.mlp(image) + features

        if not self.no_imgnorm:
            img = l2norm(features, dim=-1)
            # calculate weights
            row_global = torch.mean(img, 1)
            weights = self.v_global_w(img, row_global)

        if self.training:
            # calculate attention matrix
            img_emb = features  # V
            features_external = self.linear1(features)  # Q

            # mask1
            rand_list_1 = torch.rand(features.size(0), features.size(1)).to(features.device)
            mask1 = (rand_list_1 >= 0.2).unsqueeze(-1)
            features_external1 = features_external.masked_fill(mask1 == 0, -10000)
            features_k_softmax1 = nn.Softmax(dim=1)(
                features_external1 - torch.max(features_external1, dim=1)[0].unsqueeze(1))
            attn1 = features_k_softmax1.masked_fill(mask1 == 0, 0)
            score1 = attn1 * img_emb
            # feature_img = score * weight
            feature_img_weight = (weights.unsqueeze(2) * score1).sum(dim=1)
            feature_img1 = l2norm(feature_img_weight, dim=-1)

            # mask2
            rand_list_2 = torch.rand(features.size(0), features.size(1)).to(features.device)
            mask2 = (rand_list_2 >= 0.2).unsqueeze(-1)
            features_external2 = features_external.masked_fill(mask2 == 0, -10000)
            features_k_softmax2 = nn.Softmax(dim=1)(
                features_external2 - torch.max(features_external2, dim=1)[0].unsqueeze(1))
            attn2 = features_k_softmax2.masked_fill(mask2 == 0, 0)
            score2 = attn2 * img_emb
            # feature_img = score * weight
            feature_img_weight = (weights.unsqueeze(2) * score2).sum(dim=1)
            feature_img2 = l2norm(feature_img_weight, dim=-1)

            feature_img = torch.cat((feature_img1.unsqueeze(1), feature_img2.unsqueeze(1)), dim=1).reshape(-1,img_emb.size(-1))

        else:
            img_emb = features
            features_in = self.linear1(features)
            attn = nn.Softmax(dim=1)(features_in - torch.max(features_in, dim=1)[0].unsqueeze(1))
            score = attn * img_emb
            feature_img = (weights.unsqueeze(2) * score).sum(dim=1)

        if not self.no_imgnorm:
            feature_img = l2norm(feature_img, dim=-1)

        return feature_img

class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(num_region),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        return weights

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)
        features = self.image_encoder(base_features)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('')

        self.linear = nn.Linear(768, embed_size)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()
        self.t_global_w = TextSA(embed_size, 0.4)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]

        cap_emb = self.linear(bert_emb)

        cap_emb = self.dropout(cap_emb)

        # calculate weight
        cap = l2norm(cap_emb, dim=-1)
        weights = self.t_global_w(cap, lengths).cuda()

        # calculate attention
        max_len = int(lengths.max())
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        cap_emb = cap_emb[:, :int(lengths.max()), :]

        features_in = self.linear1(cap_emb)
        features_in = features_in.masked_fill(mask == 0, -10000)
        features_k_softmax = nn.Softmax(dim=1)(
            features_in - torch.max(features_in, dim=1)[0].unsqueeze(1))
        # attn
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        score = attn * cap_emb
        feature_cap = (weights.unsqueeze(2) * score).sum(dim=1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1)

        return feature_cap

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, lengths):
        n_caption = local.size(0)
        max_length = int(max(lengths))
        weight = torch.zeros((local.size(0), max_length))   # (batch_size,max_lengths)
        for i in range (n_caption):
            # get the i-th sentence
            n_word = int(lengths[i])
            cap_i = local[i, :n_word, :].unsqueeze(0)
            # enhance i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            # cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # compute embedding of local words and raw global text
            l_emb = self.embedding_local(cap_i)
            g_emb = self.embedding_global(cap_ave_i)

            # compute the normalized weights
            g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
            common = l_emb.mul(g_emb)
            weights = self.embedding_common(common).squeeze(2)
            weights_i = self.softmax(weights)

            if max_length > n_word:
                pad_zeros = torch.zeros(1, max_length - n_word).to(weights_i.device)
                weights_i = torch.cat((weights_i, pad_zeros), dim=1)

            weight[i] = weights_i

        return weight