import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms

from net.resnet import *
from net.branch import class_branch
from net.branch import local_branch
from rpn.anchor_target_layer import _AnchorTargetLayer
from rpn.proposal_layer import _Proposallayer

from net.l2norm import L2Norm

from net_sub.anchors import Anchors
from net_sub.loss import FocalLoss
from net_sub.utils import BBoxTransform, ClipBoxes


class Network(nn.Module):
    def __init__(self, input_size, phase='train', model='anchor', scale='single', num_classes=1):
        super(Network, self).__init__()

        resnet = resnet50(pretrained=True, receptive_keep=True)
        self.phase = phase
        # Use Detection model ('anchor' or 'CSP')
        self.detection_model = model
        # Detection scale ('single' or 'multi')
        self.detection_scale = scale

        self.n_classes = num_classes

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # self.fpn = PyramidFeatures(512, 1024, 2048)

        gh = int(input_size[0] / 16)
        gw = int(input_size[1] / 16)

        self.classification_branch = class_branch(1024, num_classes)
        self.localization_branch = local_branch(1024, gh, gw, phase)
        if self.detection_scale == 'multi':
            gh_s = int(input_size[0] / 8)
            gw_s = int(input_size[1] / 8)
            self.classification_branch_s = class_branch(512, num_classes)
            self.localization_branch_s = local_branch(512, gh_s, gw_s, phase)
            self.classification_branch_l = class_branch(2048, num_classes)
            self.localization_branch_l = local_branch(2048, gh, gw, phase)

        '''
        #use fpn
        gh_l = int(input_size[0] / 32)
        gw_l = int(input_size[1] / 32)
        self.classification_branch = class_branch(256, num_classes)
        self.localization_branch = local_branch(256, gh, gw, phase)
        self.classification_branch_s = class_branch(256, num_classes)
        self.localization_branch_s = local_branch(256, gh_s, gw_s, phase)
        self.classification_branch_l = class_branch(256, num_classes)
        self.localization_branch_l = local_branch(256, gh_l, gw_l, phase)
        '''

        if model == 'anchor':
            self.anchor_target_layer = _AnchorTargetLayer(16, np.array([4, 32]), np.array([2, ]), 0, name='middle')
            # self.soft_max = nn.Softmax(1)
            self.proposal_layer = _Proposallayer(16, np.array([4, 8]), np.array([2, ]))
            if self.detection_scale == 'multi':
                self.anchor_target_layer_s = _AnchorTargetLayer(16, np.array([2, 16]), np.array([2, ]), 0, name='small')
                self.anchor_target_layer_l = _AnchorTargetLayer(16, np.array([8, 64]), np.array([2, ]), 0, name='large')
                self.proposal_layer_s = _Proposallayer(4, np.array([1, 2]), np.array([2, ]))
                self.proposal_layer_l = _Proposallayer(32, np.array([16, 32]), np.array([2, ]))
        elif model == 'retina':
            self.anchors = Anchors()
            self.focalLoss = FocalLoss()
            self.regressBoxes = BBoxTransform()
            self.clipBoxes = ClipBoxes()
        elif model == 'csp':
            self.p3c = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
            self.p4c = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0)
            self.p5c = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0)

            self.p3c_l2 = L2Norm(128, 10)
            self.p4c_l2 = L2Norm(128, 10)
            self.p5c_l2 = L2Norm(128, 10)

            self.p3os = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
            self.p4os = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0)
            self.p5os = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0)

            self.p3os_l2 = L2Norm(128, 10)
            self.p4os_l2 = L2Norm(128, 10)
            self.p5os_l2 = L2Norm(128, 10)

            self.featc = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.featc_bn = nn.BatchNorm2d(128, momentum=0.01)
            self.featc_act = nn.ReLU(inplace=True)

            self.feat = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.feat_bn = nn.BatchNorm2d(128, momentum=0.01)
            self.feat_act = nn.ReLU(inplace=True)

            self.pos_conv = nn.Conv2d(128, num_classes, kernel_size=1)
            self.reg_conv = nn.Conv2d(128, 1, kernel_size=1)
            self.off_conv = nn.Conv2d(128, 2, kernel_size=1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, input, gt_boxes=None, im_info=None):
        ## Feature extractor ##
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # x_2, x_3, x_4 = self.fpn(x_2, x_3, x_4)

        ## bramches ##
        x_cls, x_q = self.classification_branch(x_3)
        x_obj = self.localization_branch(x_3, x_q, 'scale2')

        if self.detection_scale == 'multi':
            x_cls_s, x_q_s = self.classification_branch_s(x_2)
            x_obj_s = self.localization_branch_s(x_2, x_q_s, 'scale1')
            x_cls_l, x_q_l = self.classification_branch_l(x_4)
            x_obj_l = self.localization_branch_l(x_4, x_q_l, 'scale3')

        ##################################################################################

        if self.detection_model == 'anchor':
            if self.phase == 'train':

                labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = self.anchor_target_layer(x_cls, gt_boxes, im_info)
                if self.detection_scale == 'single':
                    return [x_cls, x_obj], [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
                elif self.detection_scale == 'multi':
                    labels_s, bbox_targets_s, bbox_inside_weights_s, bbox_outside_weights_s = self.anchor_target_layer_s(x_cls_s, gt_boxes, im_info)
                    labels_l, bbox_targets_l, bbox_inside_weights_l, bbox_outside_weights_l = self.anchor_target_layer_l(x_cls_l, gt_boxes, im_info)
                    return [[x_cls_s, x_obj_s], [x_cls, x_obj], [x_cls_l, x_obj_l]], \
                        [[labels_s, bbox_targets_s, bbox_inside_weights_s, bbox_outside_weights_s],
                        [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights],
                        [labels_l, bbox_targets_l, bbox_inside_weights_l, bbox_outside_weights_l]]

            else:
                cls_score_reshape = self.reshape(x_cls, 2)
                # softmax
                cls_prob_output = self.soft_max(cls_score_reshape)
                # reshape from (batch,2,2*H,W) back to (batch,4,h,w)
                cls_prob_reshape = self.reshape(cls_prob_output, 4)
                # roi has shape of (batch, top_k, 5)
                # where (batch, top_k, 4) is cls score and
                # (batch, top_k, 0:4) is bbox coordinated
                roi_1 = self.proposal_layer(cls_prob_reshape, x_obj, im_info)
                if self.detection_scale == 'single':
                    return roi_1

                elif self.detection_scale == 'multi':
                    cls_score_reshape_s = self.reshape(x_cls_s, 2)
                    cls_score_reshape_l = self.reshape(x_cls_l, 2)

                    cls_prob_output_s = self.soft_max(cls_score_reshape_s)
                    cls_prob_output_l = self.soft_max(cls_score_reshape_l)

                    cls_prob_reshape_s = self.reshape(cls_prob_output_s, 4)
                    cls_prob_reshape_l = self.reshape(cls_prob_output_l, 4)

                    roi_2 = self.proposal_layer_s(cls_prob_reshape_s, x_obj_s, im_info)
                    roi_3 = self.proposal_layer_l(cls_prob_reshape_l, x_obj_l, im_info)

                    roi = torch.cat((roi_1, roi_2, roi_3), dim=1)
                    return roi

        ##################################################################################

        elif self.detection_model == 'retina':
            anchors = self.anchors(input)
            x_cls_1 = x_cls.permute(0, 2, 3, 1)
            batch_size, height, width, channels = x_cls_1.shape
            out2 = x_cls_1.view(batch_size, width, height, 3, self.n_classes)
            classification = out2.contiguous().view(x.shape[0], -1, self.n_classes)
            x_obj_1 = x_obj.permute(0, 2, 3, 1)
            regression = x_obj_1.contiguous().view(x_obj_1.shape[0], -1, 4)
            if self.detection_scale == 'single':
                classification_cat = classification
                regression_cat = regression

            elif self.detection_scale == 'multi':
                # 3scale detection
                x_cls_1_s = x_cls_s.permute(0, 2, 3, 1)
                batch_size, height_s, width_s, channels = x_cls_1_s.shape
                out2_s = x_cls_1_s.view(batch_size, width_s, height_s, 3, self.n_classes)
                classification_s = out2_s.contiguous().view(x.shape[0], -1, self.n_classes)
                x_obj_1_s = x_obj_s.permute(0, 2, 3, 1)
                regression_s = x_obj_1_s.contiguous().view(x_obj_1_s.shape[0], -1, 4)

                x_cls_1_l = x_cls_l.permute(0, 2, 3, 1)
                batch_size, height_l, width_l, channels = x_cls_1_l.shape
                out2_l = x_cls_1_l.view(batch_size, width_l, height_l, 3, self.n_classes)
                classification_l = out2_l.contiguous().view(x.shape[0], -1, self.n_classes)
                x_obj_1_l = x_obj_l.permute(0, 2, 3, 1)
                regression_l = x_obj_1_l.contiguous().view(x_obj_1_l.shape[0], -1, 4)

                classification_cat = torch.cat([classification, classification_l, classification_s], dim=1)
                regression_cat = torch.cat([regression, regression_l, regression_s], dim=1)

            if self.phase == 'train':
                return self.focalLoss(classification_cat, regression_cat, anchors, gt_boxes)
            else:
                transformed_anchors = self.regressBoxes(anchors, regression_cat)
                transformed_anchors = self.clipBoxes(transformed_anchors, input)

                finalResult = [[], [], []]

                finalScores = torch.Tensor([])
                finalAnchorBoxesIndexes = torch.Tensor([]).long()
                finalAnchorBoxesCoordinates = torch.Tensor([])
                # score_number = torch.tensor(range((height * width) + (height_s * width_s) + (height_l * width_l) * 3))
                # score_number = score_number.view(width, height, -1)

                if torch.cuda.is_available():
                    finalScores = finalScores.cuda()
                    finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                    finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

                for i in range(classification.shape[2]):
                    # import pdb
                    # pdb.set_trace()
                    scores = torch.squeeze(classification_cat[:, :, i])
                    scores_over_thresh = (scores > 0.05)
                    if scores_over_thresh.sum() == 0:
                        # no boxes to NMS, just continue
                        continue

                    scores = scores[scores_over_thresh]
                    # over_thresh_number = score_number[scores_over_thresh]
                    anchorBoxes = torch.squeeze(transformed_anchors)
                    anchorBoxes = anchorBoxes[scores_over_thresh]
                    anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
                    # print (over_thresh_number[anchors_nms_idx])

                    finalResult[0].extend(scores[anchors_nms_idx])
                    finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                    finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                    finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                    finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                    if torch.cuda.is_available():
                        finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                    finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                    finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

                return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]

        ##################################################################################

        elif self.detection_model == 'csp':
            p3c = self.p3c(x_cls_s)
            p3c = self.p3c_l2(p3c)
            p4c = self.p4c(x_cls)
            p4c = self.p4c_l2(p4c)
            p5c = self.p5c(x_cls_l)
            p5c = self.p5c_l2(p5c)

            p3os = self.p3os(x_obj_s)
            p3os = self.p3os_l2(p3os)
            p4os = self.p4os(x_obj)
            p4os = self.p4os_l2(p4os)
            p5os = self.p5os(x_obj_l)
            p5os = self.p5os_l2(p5os)
            # print(p3c.size(),p4c.size(),p5c.size())

            cat_c = torch.cat([p3c, p4c, p5c], dim=1)
            # cat = cat + cat_s
            feat_c = self.featc(cat_c)
            feat_c = self.featc_bn(feat_c)
            feat_c = self.featc_act(feat_c)

            cat_os = torch.cat([p3os, p4os, p5os], dim=1)
            # cat = cat + cat_s
            feat = self.feat(cat_os)
            feat = self.feat_bn(feat)
            feat = self.feat_act(feat)

            o_cls = self.pos_conv(feat_c)
            o_cls = torch.sigmoid(o_cls)
            # x_cls = F.softmax(feat, 1)
            o_reg = self.reg_conv(feat)
            o_off = self.off_conv(feat)
            return [o_cls, o_reg, o_off]

        ##################################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid2(x):
    return np.exp(x) / (np.exp(x) + 1.0)


def max_a(x):
    return x / max(x)


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, C3, C4, C5):
        # C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return P3_x, P4_x, P5_x

