import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

class Local_attention(nn.Module):
    def __init__(self, in_channels, local_scale):
        super(Local_attention, self).__init__()
        #self.conv7x7 = nn.Conv2d(in_channels, 64, kernel_size=7, dilation=2, padding=6)
        #self.conv1x1 = nn.Conv2d(128, local_scale ** 2, kernel_size = 1)
        #self.padding = nn.ZeroPad2d(int(local_scale/2))
        #self.conv1x1 = nn.Conv2d(64, 1, kernel_size = 1)
        #self.ch_dic = nn.Conv3d(in_channels, 128, kernel_size=1)
        self.conv3d3x3 = nn.Conv3d(in_channels, 32, kernel_size=(1,3,3),stride=1,padding=(0,1,1))
        #self.conv3d3x3 = nn.Conv3d(in_channels, 64, kernel_size=(1,3,3),stride=1,padding=(0,1,1))
        self.conv3d1x1 = nn.Conv3d(32,1,kernel_size=1)
        #self.conv3d1x1 = nn.Conv3d(64,1,kernel_size=1)
        self.conv3d9x9x1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, local_scale, local_scale))


    def forward(self, input, local_scale, type):
        x = input
        s = x.size()
        if type == 'sliding_window':
            pad_size = int(local_scale/2)
            x_unf = F.unfold(x, kernel_size=(s[2] - local_scale + pad_size * 2 + 1, (s[3] - local_scale + pad_size * 2 + 1)), padding=pad_size)
            #x_unf = F.unfold(x, kernel_size=(local_scale, local_scale), padding=pad_size)
            local_x = x_unf.reshape(s[0], s[1], -1, local_scale, local_scale)
            #print(local_x.size())
            #kernel = self.ch_dic(local_x)
            kernel = self.conv3d3x3(local_x)
            #print(kernel.size())
            kernel = self.conv3d1x1(kernel)
            #att = F.softmax(kernel, 1)
            att = F.sigmoid(kernel)
            att_mp = att
            att_x = local_x * att + local_x
            out_x = self.conv3d9x9x1(att_x)
            output = out_x.reshape(s[0], s[1], s[2], s[3])
        elif type == 'fixed_area':
            h = int(s[2] / local_scale)
            w = int(s[3] / local_scale)
            att_mp = torch.zeros(s[0], (h)*local_scale*(w)*local_scale, local_scale, local_scale, requires_grad=False).type(torch.FloatTensor)
            output = torch.zeros(s).type(torch.FloatTensor)
            output = output.to(device)
            local_x = torch.zeros(h,w,s[0],s[1],local_scale,local_scale).type(torch.FloatTensor)
            local_x = local_x.to(device)
            mask = torch.zeros(x.size(),requires_grad=False).type(torch.FloatTensor)
            mask = mask.to(device)
            for i in range(h):
                for j in range(w):
                    mask[:,:,:,:] = 0
                    mask[:, :, i*local_scale:(i+1)*local_scale, j*local_scale:(j+1)*local_scale] = 1
                    local_x = x[mask != 0]
                    local_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    local_x = local_x.reshape(1, s[1], local_scale, local_scale)
                    size = local_x.size()
                    kernel = self.conv7x7(local_x)
                    kernel = self.conv1x1(kernel)
                    att = F.softmax(kernel, 1)
                    att_mp[:,(i+j)*local_scale*local_scale:(i+j+1)*local_scale*local_scale,:,:] = att
                    att_m = att.reshape(size[0], 1, size[2] * size[3], local_scale ** 2)
                    mul = F.unfold(local_x, kernel_size=[local_scale, local_scale], dilation=[2, 2], padding=local_scale-1)
                    mul = mul.reshape(size[0], size[1], size[2] * size[3], -1)
                    mul = torch.mul(mul, att_m)
                    mul = torch.sum(mul, dim=3)
                    local_x = mul.reshape(size[0], size[1], size[2], size[3])
                    output[:,:,local_scale*i:local_scale*(i+1),local_scale*j:local_scale*(j+1)] = local_x
        else:
            raise ValueError("not found LAP type: {}".format(type))

        return output, att_mp
