import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

class UTO(nn.Module):
    def __init__(self, opt):
        super(UTO, self).__init__()
        self.opt = opt
        self.l_alpha = opt.mu
        self.l_ep = opt.gama

    def forward(self, im, s):
        bsize = im.size()[0]
        scores = get_sim(im, s)

        tmp = torch.eye(bsize).cuda()
        s_diag = tmp * scores
        scores_ = scores - s_diag
        S_ = torch.exp(self.l_alpha * (scores_ - self.l_ep))

        loss_diag_1 = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum(
            torch.log(1 + S_.sum(0)) / self.l_alpha + torch.log(1 + S_.sum(1)) / self.l_alpha + loss_diag_1) / bsize

        return loss

    def moco_forward(self, v_q, t_k, t_q, v_k, v_queue, t_queue):
        # v positive logits: Nx1
        v_pos = torch.einsum("nc,nc->n", [v_q, t_k]).unsqueeze(-1)
        # v negative logits: NxK
        t_queue = t_queue.clone().detach()
        v_neg = torch.einsum("nc,ck->nk", [v_q, t_queue])

        # # t positive logits: Nx1
        t_pos = torch.einsum("nc,nc->n", [t_q, v_k]).unsqueeze(-1)
        # t negative logits: NxK
        v_queue = v_queue.clone().detach()
        t_neg = torch.einsum("nc,ck->nk", [t_q, v_queue])

        v_pos_diag = torch.diag_embed(v_pos.squeeze(-1))
        v_bsize = v_pos_diag.size()[0]
        v_loss_diag = - torch.log(1 + F.relu(v_pos_diag.sum(0)))
        v_S_ = torch.exp((v_neg - self.l_ep) * self.l_alpha)
        v_S_T = v_S_.T
        v_loss = torch.sum(torch.log(1 + v_S_T.sum(0)) / self.l_alpha + v_loss_diag) / v_bsize

        t_pos_diag = torch.diag_embed(t_pos.squeeze(-1))
        t_bsize = t_pos_diag.size()[0]
        t_loss_diag = - torch.log(1 + F.relu(t_pos_diag.sum(0)))
        t_S_ = torch.exp((t_neg - self.l_ep) * self.l_alpha)
        t_S_T = t_S_.T
        t_loss = torch.sum(torch.log(1 + t_S_T.sum(0)) / self.l_alpha + t_loss_diag) / t_bsize

        return v_loss + t_loss
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)  # bk,b: (256,256)
     
        hardnum = self.opt.hardnum
        mm_a = (torch.arange(scores.size(0)) // self.opt.hardnum + 1) * self.opt.hardnum
        mask_a = torch.arange(im.size(0)).view(im.size(0), 1).expand_as(scores)
        mask1 = (mask_a < mm_a.long())
        mask = mask1 * mask1.t()
        if torch.cuda.is_available():
            I = mask.cuda()    # (256,256)

        # caption retrieval
        scores_inner = torch.masked_select(scores, I).reshape(scores.size(0)//hardnum, hardnum, hardnum)

        scores_image = scores_inner.min(dim=2)[0].reshape((-1, 1))  # (256,1)
        cost_s = (self.margin + scores - scores_image.view(im.size(0), 1).expand_as(scores)).clamp(min=0)

        # image retrieval
        scores_caption = scores_inner.min(dim=1)[0].reshape((1,-1))
        cost_im = (self.margin + scores - scores_caption.view(1,s.size(0)).expand_as(scores)).clamp(min=0)

        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if self.max_violation:
            cost_im = cost_im.max(0)[0]
            cost_s = cost_s.max(1)[0]
            
        cost_im =cost_im.sum()    
        cost_s =cost_s.sum()
        return cost_im, cost_s

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X