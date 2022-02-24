import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import math
import torch.nn.init as init

def visualize(img, name):
    npimgs = img
    for i in range(len(npimgs)):
        npimg = npimgs[i].cpu().detach().numpy()
        filename = name + "_" + str(i) + ".png"
        plt.figure()
        plt.imshow(npimg)
        plt.show()
        plt.colorbar()
        plt.imsave("data/output/attmap/" + filename, npimg)
        plt.close()


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """

    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage

class class_branch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(class_branch, self).__init__()

        #abn_convtype = config_model['ABN_CONVTYPE']
        abn_convtype = 'simple'
        if abn_convtype == 'simple':
            last_out_ch = 256
            self.conv_layer1 = nn.Conv2d(in_channels=in_channels,
                      out_channels=512, kernel_size=3, stride=2,
                      padding=(3 - 1) // 2, bias=False)
            self.norm_layer1 = nn.BatchNorm2d(512)
            self.relu_layer1 = nn.LeakyReLU(0.1)
            self.conv_layer2 = nn.Conv2d(in_channels=512,
                                         out_channels=last_out_ch, kernel_size=3, stride=1,
                                         padding=(3 - 1) // 2, bias=False)
            self.norm_layer2 = nn.BatchNorm2d(last_out_ch)
            self.relu_layer2 = nn.LeakyReLU(0.1)
        else:
            raise ValueError('bad ABN_CONVTYPE:{}'.format(self.abn_convtype))
        '''
        elif abn_convtype == 'complex':
            if layer_no == 0:
                # attention branch for 1st yolo branch (scale 3)
                mlist.append(resblock(ch=1024, nblocks=4))  # 0_0
                mlist.append(
                    resblock(ch=1024, nblocks=2, shortcut=False))  # 0_1
                mlist.append(add_conv(in_ch=1024, out_ch=512,
                                      ksize=1, stride=1))  # 0_2
                mlist.append(add_conv(in_ch=512, out_ch=1024,
                                      ksize=3, stride=1))  # 0_3
                last_out_ch = 1024  # layer a32(13) outputs 1024 channels
            elif layer_no == 1:
                # attention branch for 2nd yolo branch (scale 2)
                mlist.append(add_conv(in_ch=768, out_ch=256,
                                      ksize=1, stride=1))  # 1_0
                mlist.append(add_conv(in_ch=256, out_ch=512,
                                      ksize=3, stride=1))  # 1_1
                mlist.append(
                    resblock(ch=512, nblocks=1, shortcut=False))  # 1_2
                mlist.append(add_conv(in_ch=512, out_ch=256,
                                      ksize=1, stride=1))  # 1_3
                last_out_ch = 256   # layer a44(20) outputs 256 channels
            elif layer_no == 2:
                # attention branch for 3rd yolo branch (scale 1)
                mlist.append(add_conv(in_ch=384, out_ch=128,
                                      ksize=1, stride=1))  # 2_0
                mlist.append(add_conv(in_ch=128, out_ch=256,
                                      ksize=3, stride=1))  # 2_1
                mlist.append(
                    resblock(ch=256, nblocks=2, shortcut=False))  # 2_3
                last_out_ch = 256   # layer a55(27) outputs 256 channels
        '''

        self.abn_target = 'obj_prob' # config_model['ABN_TARGET']
        self.use_old_abn = True

        # attention branch common
        n_classes = num_classes+1

        self.conv1x1_class = nn.Conv2d(in_channels=last_out_ch,
                                       out_channels=(n_classes) * 3, kernel_size=1, stride=1, bias=False)
        #self.softmax_class = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

        self.conv1x1_query = nn.Conv2d(in_channels=(n_classes) * 3,
                                       out_channels=128, kernel_size=1, stride=1, bias=False)
        '''
        #cspmodel
        self.conv1x1_class = nn.Conv2d(in_channels=last_out_ch,
                                       out_channels=128, kernel_size=1, stride=1, bias=False)
        self.conv1x1_query = nn.Conv2d(in_channels=128,
                                       out_channels=128, kernel_size=1, stride=1, bias=False)
        '''

    def forward(self, x, train=False):
        x = self.conv_layer1(x)
        x = self.norm_layer1(x)
        x = self.relu_layer1(x)
        x = self.conv_layer2(x)
        x = self.norm_layer2(x)
        out = self.relu_layer2(x)

        out_class = self.conv1x1_class(out)

        out_query = self.conv1x1_query(out_class)

        out_class = self.sigmoid(out_class) # commentout if use cspmodel

        return out_class, out_query

class local_branch(nn.Module):
    def __init__(self, in_channels, gh, gw, phase):
        super(local_branch, self).__init__()
        self.phase = phase

        self.conv3d1x1 = nn.Conv3d(128, 1, kernel_size=1)

        self.conv1x1_key = nn.Conv2d(in_channels=in_channels,
                                       out_channels=128, kernel_size=1, stride=1, bias=False)

        self.conv1x1_obj = nn.Conv2d(in_channels=gh * gw,
                                     out_channels=4 * 3, kernel_size=1, stride=1, bias=False)

        #self.conv1x1_csp = nn.Conv2d(in_channels=gh * gw,
        #                             out_channels=128, kernel_size=1, stride=1, bias=False)

    def forward(self, k, q, name):
        ###  grid-wise-attention  ###
        key = self.conv1x1_key(k)
        #value = self.conv1x1_key(k)
        q_s = q.size()
        k_s = key.size()
        query = q.view(q_s[0], q_s[1], q_s[2]*q_s[3])
        att = torch.einsum('b d i, b d h w -> b i h w', query, key)

        if self.phase == 'test':
            visualize(att[0], name +'att')

        att_v = att.view(q_s[0], 1, q_s[2]*q_s[3], k_s[2], k_s[3])
        val_v = key.view(k_s[0], k_s[1], 1, k_s[2], k_s[3])

        ###  localization  ###
        v_w = torch.mul(val_v, att_v) + val_v
        out = self.conv3d1x1(v_w)
        out = out.view(k_s[0], 1, q_s[2] * q_s[3], k_s[2] * k_s[3])
        out = out.permute(0, 3, 2, 1)
        out = out.view(k_s[0], k_s[2] * k_s[3], q_s[2], q_s[3])

        out = self.conv1x1_obj(out)
        #out = self.conv1x1_csp(out)
        return out



