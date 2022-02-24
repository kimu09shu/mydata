import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

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



class ABNLayer(nn.Module):
    #def __init__(self, config_model, layer_no, in_channels):
    def __init__(self, in_channels):
        super(ABNLayer, self).__init__()

        mlist = nn.ModuleList()

        #abn_convtype = config_model['ABN_CONVTYPE']
        abn_convtype = 'simple'
        if abn_convtype == 'simple':
            last_out_ch = 512
            mlist.append(add_conv(in_ch=in_channels,
                                  out_ch=512, ksize=3, stride=1))
            mlist.append(
                add_conv(in_ch=512, out_ch=last_out_ch, ksize=3, stride=1))
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
        n_classes = 1 # config_model['N_CLASSES']
        # abn common 1
        if self.use_old_abn:
            #mlist.append(nn.BatchNorm2d(last_out_ch))  # c1_0
            mlist.append(nn.Conv2d(last_out_ch, n_classes,
                                   kernel_size=1, padding=0, bias=False))  # c1_1
            mlist.append(nn.BatchNorm2d(n_classes))  # c1_2
            mlist.append(nn.ReLU(inplace=True))     # c1_3
        else:
            abn1 = nn.Sequential()
            abn1.add_module('abn1_batch_norm1',
                            nn.BatchNorm2d(last_out_ch))  # c1_0
            abn1.add_module('abn1_conv', nn.Conv2d(
                last_out_ch, n_classes, kernel_size=1, padding=0, bias=False))  # c1_1
            abn1.add_module('abn1_batch_norm2',
                            nn.BatchNorm2d(n_classes))  # c1_2
            abn1.add_module('abn1_leaky', nn.ReLU(inplace=True))  # c1_3
            mlist.append(abn1)
        # for common 1 -> 3 in training phase
        self.common_1_out_no = len(mlist) - 1
        #logger.debug('self.common_1_out_no:{}'.format(self.common_1_out_no))
        # abn common 2
        if self.use_old_abn:
            mlist.append(nn.Conv2d(n_classes, 1, kernel_size=3,
                                   padding=1, bias=False))  # c2_0
            mlist.append(nn.BatchNorm2d(1))       # c2_1
            mlist.append(nn.Sigmoid())            # c2_3
        else:
            abn2 = nn.Sequential()
            abn2.add_module('abn2_conv', nn.Conv2d(
                n_classes, 1, kernel_size=3, padding=1, bias=False))  # c2_0
            abn2.add_module('abn2_batch_norm', nn.BatchNorm2d(1))  # c2_1
            abn2.add_module('abn2_sigmoid', nn.Sigmoid())         # c2_3
            mlist.append(abn2)
        self.common_2_out_no = len(mlist) - 1  # for attention map output
        #logger.debug('self.common_2_out_no:{}'.format(self.common_2_out_no))
        if self.abn_target == 'class_prob':
            # abn common 3
            if self.use_old_abn:
                mlist.append(nn.Conv2d(n_classes, n_classes,
                                       kernel_size=1, padding=0, bias=False))  # c3_0
                mlist.append(nn.Softmax2d())          # c3_1
            else:
                abn3 = nn.Sequential()
                abn3.add_module('abn3_conv', nn.Conv2d(
                    n_classes, n_classes, kernel_size=1, padding=0, bias=False))  # c3_0
                abn3.add_module(nn.Softmax2d())          # c3_1
                mlist.append(abn3)
            self.common_3_out_no = len(mlist) - 1  # for prob map output
            #logger.debug('self.common_3_out_no:{}'.format(self.common_3_out_no))

        self.module_list = mlist
        # logger.debug('layer_no:{} self.common_1_out_no:{}'.format(layer_no, self.common_1_out_no))
        # logger.debug('layer_no:{} self.common_2_out_no:{}'.format(layer_no, self.common_2_out_no))
        # logger.debug('layer_no:{} self.common_3_out_no:{}'.format(layer_no, self.common_3_out_no))

    def forward(self, x, train=False):
        output = []
        for i, module in enumerate(self.module_list):
            x = module(x)

            if self.abn_target == 'class_prob':
                if i == self.common_1_out_no:
                    # logger.debug('i:{} store x:{} x.shape:{}'.format(i, x, x.shape))
                    common_1_out = x
                if i == self.common_3_out_no:
                    # logger.debug('i:{} train output.append(x):{} x.shape:{}'.format(i, x, x.shape))
                    output.append(x)  # prob map
                    print(output)
                    # logger.debug('i:{} x:{}'.format(i, x))

            if i == self.common_2_out_no:
                # logger.debug('i:{} output.append(x):{} x.shape:{}'.format(i, x, x.shape))
                output.append(x)  # attention map
                #print(output)
                if self.abn_target == 'class_prob':
                    if train:
                        x = common_1_out
                        # logger.debug('i:{} x:{}'.format(i, x))
                    else:
                        break

        if self.abn_target == 'class_prob':
            if not train:
                output.append(None)  # prob is None in test phase.
            #logger.debug('output: {}'.format(output))

        return output[0]
