import torch.nn as nn
import torch
import numpy as np


class cls_pos(nn.Module):
    def __init__(self):
        super(cls_pos, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])

        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]

        fore_weight = positives * (1.0-pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * ((1.0-pos_label[:, 0, :, :])**4.0) * (pos_pred[:, 0, :, :]**2.0)

        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(pos_label[:, 2, :, :])

        cls_loss = 0.01 * torch.sum(focal_weight*log_loss) / max(1.0, assigned_box)

        return cls_loss


class reg_pos(nn.Module):
    def __init__(self):
        super(reg_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :]*self.smoothl1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
        return reg_loss


class offset_pos(nn.Module):
    def __init__(self):
        super(offset_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_label[:, :2, :, :])
        off_loss = 0.1 * torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
        return off_loss

device = torch.device("cuda")#("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor
dtype_l = torch.FloatTensor
torch.autograd.set_detect_anomaly(True)

class AttLoss(nn.Module):
    def __init__(self):
        super(AttLoss, self).__init__()
        self.l1_loss = nn.L1Loss(size_average=False)
        self.l2_loss = nn.MSELoss(size_average=False)  # MSE
        self.bce_loss = nn.BCELoss(size_average=False)

        self.abn_losstype = 'l2'

    def forward(self, att_map, seg_bb):
        batchsize = att_map[0].shape[0]

        loss_prob_grid1 = 0
        loss_prob_grid2 = 0
        loss_grid1, loss_grid2 = 0, 0

        gt_area = 1

        for i in range(len(att_map)):

            att_m = att_map[i].float()

            fsize = att_m.shape[2]
            fsize1 = att_m.shape[3]

            prob_target = torch.zeros(batchsize, 1, fsize, fsize1,requires_grad=False).type(dtype_l)
            prob_target = prob_target.to(device)
            seg_bb = seg_bb.cpu().data

            nlabel = (seg_bb.sum(dim=2) > 0).sum(dim=1)  # number of objects

            truth_xt_gam = seg_bb[:, :, 0] * fsize1 / 2048
            truth_yt_gam = seg_bb[:, :, 1] * fsize / 1024
            truth_xu_gam = seg_bb[:, :, 2] * fsize1 / 2048
            truth_yu_gam = seg_bb[:, :, 3] * fsize / 1024

            for b in range(batchsize):
                n = int(nlabel[b])
                if n > 0:
                    # Loss(true_k ,gt_k)
                    for ti in range(n):
                        tmp_right = int(truth_xu_gam[b,ti])
                        tmp_bottom = int(truth_yu_gam[b,ti])

                        tmp_left = int(truth_xt_gam[b,ti])# i_gam #- tgt_width // 2
                        tmp_top = int(truth_yt_gam[b,ti])# j_gam #- tgt_height // 2

                        gt_area+=(tmp_bottom - tmp_top) * (tmp_right - tmp_left)

                        prob_target[b, 0, tmp_top:tmp_bottom, tmp_left: tmp_right] = 1.0

            att_m_gt = att_m * prob_target
            if self.abn_losstype == 'l2':
                loss_prob1 = self.l1_loss(att_m_gt, prob_target)  # MSE
            elif self.abn_losstype == 'bce':
                loss_prob1 = self.bce_loss(att_m_gt, prob_target)  # BCE
            else:
                raise ValueError('bad ABN_LOSSTYPE:{}'.format(self.abn_losstype))

            if self.abn_losstype == 'l2':
                loss_prob2 = self.l2_loss(att_m, prob_target)  # MSE
            elif self.abn_losstype == 'bce':
                loss_prob2 = self.bce_loss(att_m, prob_target)  # BCE
            else:
                raise ValueError('bad ABN_LOSSTYPE:{}'.format(self.abn_losstype))
            loss_prob_grid1 += loss_prob1 / gt_area
            loss_prob_grid2 += loss_prob2 / (fsize * fsize1)
            loss_grid1 = loss_prob_grid1
            loss_grid2 = loss_prob_grid2

        return loss_grid1 ,loss_grid2

