import torch.nn as nn
import torch
import numpy as np

import math

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

        '''
        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]

        fore_weight = positives * (1.0 - pos_pred[:, 1, :, :]) ** 2 * torch.log(pos_pred[:, 1, :, :])
        back_weight = negatives * ((1.0 - pos_label[:, 0, :, :]) ** 4) * (1 - pos_pred[:, 0, :, :]) ** 2 * torch.log(
            pos_pred[:, 0, :, :])

        positives_2 = pos_label[:, 3, :, :]
        negatives_2 = pos_label[:, 1, :, :] - pos_label[:, 3, :, :]

        fore_weight_2 = positives_2 * (1.0 - pos_pred[:, 2, :, :]) ** 2 * torch.log(pos_pred[:, 2, :, :])
        back_weight_2 = negatives_2 * ((1.0 - pos_label[:, 0, :, :]) ** 4) * (1 - pos_pred[:, 0, :, :]) ** 2 * torch.log(
            pos_pred[:, 0, :, :])

        positives_3 = pos_label[:, 4, :, :]
        negatives_3 = pos_label[:, 1, :, :] - pos_label[:, 4, :, :]

        fore_weight_3 = positives_3 * (1.0 - pos_pred[:, 3, :, :]) ** 2 * torch.log(pos_pred[:, 3, :, :])
        back_weight_3 = negatives_3 * ((1.0 - pos_label[:, 0, :, :]) ** 4) * (1 - pos_pred[:, 0, :, :]) ** 2 * torch.log(
            pos_pred[:, 0, :, :])

        focal_weight = fore_weight + back_weight + fore_weight_2 + back_weight_2 + fore_weight_3 + back_weight_3
        assigned_box = torch.sum(pos_label[:, 2:5, :, :])

        '''

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

        loss_prob1 = np.zeros(1)
        gt_area = 1

        for i in range(len(att_map)):

            att_m = att_map[i].float()
            att_m_flag = torch.zeros(att_m.size(),requires_grad=False).type(dtype_l)
            att_m_flag = att_m_flag.to(device)

            fsize = att_m.shape[2]
            fsize1 = att_m.shape[3]

            prob_target = torch.zeros(batchsize, 1, fsize, fsize1,requires_grad=False).type(dtype_l)
            prob_target = prob_target.to(device)
            seg_bb = seg_bb.cpu().data

            #nlabel =np.zeros(batchsize)
            nlabel = (seg_bb.sum(dim=2) > 0).sum(dim=1)  # number of objects

            truth_xt_gam = seg_bb[:, :, 0] * fsize1 / 2048
            truth_yt_gam = seg_bb[:, :, 1] * fsize / 1024
            truth_xu_gam = seg_bb[:, :, 2] * fsize1 / 2048
            truth_yu_gam = seg_bb[:, :, 3] * fsize / 1024
            #print(truth_xt_gam,truth_yt_gam,truth_xu_gam,truth_yu_gam)
            truth_i_gam = truth_xt_gam.to(torch.int16).numpy()
            truth_j_gam = truth_yt_gam.to(torch.int16).numpy()

            for b in range(batchsize):
                '''
                if len(seg_bb[b]) > 0:
                    nlabel = (seg_bb.sum(dim=2) > 0).sum(dim=1)  # number of objects
                '''

                n = int(nlabel[b])
                if n > 0:
                    #loss_prob1 = torch.zeros(1,requires_grad=False).type(dtype)
                    #break;
                #else:
                    #truth_xt_gam = seg_bb[b, :, 0] * fsize1 / 2048
                    #truth_yt_gam = seg_bb[b, :, 1] * fsize / 1024
                    #truth_xu_gam = seg_bb[b, :, 2] * fsize1 / 2048
                    #truth_yu_gam = seg_bb[b, :, 3] * fsize / 1024
                    '''
                    truth_gam = torch.Tensor(np.zeros((n, 4)))
                    truth_gam[:n, 0] = truth_xt_gam[b, :n]
                    truth_gam[:n, 1] = truth_yt_gam[b, :n]
                    truth_gam[:n, 2] = truth_xu_gam[b, :n]
                    truth_gam[:n, 3] = truth_yu_gam[b, :n]
                    truth_gam_xt = truth_i_gam[b, :n]
                    truth_gam_yt = truth_j_gam[b, :n]
                    '''

                    # Loss(true_k ,gt_k)
                    for ti in range(n):
                        #i_gam, j_gam = truth_gam_xt[ti], truth_gam_yt[ti]  # addition

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


def reshape(x, d):
    input_shape = x.size()
    x = x.view(
        input_shape[0],
        int(d),
        int(float(input_shape[1] * input_shape[2]) / float(d)),
        input_shape[3]
    )
    return x

def clsLoss(cls_score, labels, batch):
    batch_size = batch
    # reshape from (batch,4,h,w) to (batch,2,2*h,w)
    cls_score_reshape = reshape(cls_score, 2)
    sm = nn.Softmax(1)
    cls_score_reshape = sm(cls_score_reshape)

    # reshape from (batch, 2, 2*h,w) to (batch, 2*h*w,2)
    cls_score = cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
    # reshape from (batch,1,2*H,W) to (batch, 2*h*w)
    target_labels = labels.view(batch_size, -1)

    # reshape to (N,C) for cross_entropy loss
    cls_score = cls_score.view(-1, 2)

    # reshape to (N)
    target_labels = target_labels.view(-1).long()

    # compute bbox classification loss
    cls_loss = nn.functional.cross_entropy(cls_score, target_labels, ignore_index=-1)

    return cls_loss


def boxLoss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box