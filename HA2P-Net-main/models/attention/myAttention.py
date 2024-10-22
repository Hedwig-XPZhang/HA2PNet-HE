import torch
from torch import nn
from models.base_ops.involution import Involution
from torch.nn import init
import torch.nn.functional as F
from models.attention.ife import CNN_qulv,CNN_Entropy

class CFM(nn.Module):

    def __init__(self, channels=64 ,r=16):
        super(CFM, self).__init__()


        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channels, channels // r, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // r, channels, 1, bias=False),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels// r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels // r, kernel_size=3, stride=1, padding=1, groups=channels // r),
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        xl = self.avgpool(x)
        xl = self.se(xl)
        xg = self.dw(x)
        # print(xl.shape)
        res = xl + xg
        # print(res.shape)

        wei = self.sigmoid(res)

        return x*wei + x

class CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=16):
        super(CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            # nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        # print(xl.shape)
        xg = self.global_att(x)
        # print(xg.shape)
        res = xl + xg  #b*c*h*w
        # wei = self.sigmoid(xlg)
        # print(res.shape)

        return res
class SAM(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=4, dia_val=4):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):#Involution(kernel_size=7, channels= c_list[0], stride=1)
            # nn.Conv2d(kernel_size=3, in_channels=channel // reduction,out_channels=channel // reduction, padding=dia_val, dilation=dia_val)
            self.sa.add_module('conv_%d' % i, Involution(kernel_size=7, channels= channel//reduction, stride=1))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)#b,1,h,w
        return res

class SAM2(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)  #b,1,h,w
        out = self.sigmoid(output)
        return out
class CAM2(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        # output = max_out + avg_out
        # print(output.shape)
        return output

class DAFM(nn.Module):
    '''
    多特征融合
    '''

    def __init__(self, channels=64,kernel_size=7):
        super(DAFM, self).__init__()
        # self.ca = CAM(channels=channels, r=r)
        self.ca = CAM2(channel=channels)
        # self.sa = SAM(channel=channels)
        self.sa = SAM2(kernel_size=kernel_size)

        # self.ca1 = CAM2(channel=channels)
        # self.sa1 = SAM2(kernel_size=kernel_size)
        # self.sigmoid = nn.Sigmoid()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x, residual):
        x_in = x + residual

########【并行】##############################
        # sa_out = self.sa(x_in)
        # ca_out = self.ca(x_in)
        # weight = self.sigmoid(sa_out * ca_out)
        #
        # out = x * weight + residual * (1 - weight) + x_in
########【串行】##############################
        wei_c = self.ca(x_in)
        res = x * wei_c + residual * (1 - wei_c)
        wei_s = self.sa(res)
        out = x * wei_c * wei_s + residual * (1 - wei_c) * (1 - wei_s)

        return out

################T-am
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, a, b):
        x = a + b
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return a * scale + b * (1 - scale)


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, a, b):
        x = a + b
        # x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        a_perm1 = a.permute(0, 2, 1, 3).contiguous()
        b_perm1 = b.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(a_perm1,b_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        # x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        a_perm2 = a.permute(0, 3, 2, 1).contiguous()
        b_perm2 = b.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(a_perm2,b_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_out = self.hw(a,b)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)

        return x_out

##############G-am
class GAM_Attention(nn.Module):
    def __init__(self, in_channels=256, rate=4):
        super(GAM_Attention, self).__init__()
        inner = in_channels // rate
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, inner),
            nn.ReLU(inplace=True),
            nn.Linear(inner, in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, inner, kernel_size=7, padding=3),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )
        self.sigmid = nn.Sigmoid()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, y,z):
        x = y + z
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        wei_c = self.sigmid(x_channel_att)
        x1 = y * wei_c + z * (1 - wei_c)
        # x1 = x * wei_c

        x_spatial_att = self.spatial_attention(x1)
        wei_s = self.sigmid(x_spatial_att)
        out = y * wei_c * wei_s + z * (1 - wei_c) * (1 - wei_s)
        # out = x1 * wei_s
        return out



###########ourFFM#########
class HAFFP(nn.Module):
    def __init__(self,channels=[48, 64, 160, 256],out_channal=256,ife = 'c'):
        super(HAFFP, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(channels[3], out_channal, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(out_channal, out_channal, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(out_channal, out_channal, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(out_channal, out_channal, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(channels[2], out_channal, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(channels[1], out_channal, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(channels[0], out_channal, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(in_channels=out_channal * 4, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.fusion1 = TripletAttention()
        self.fusion2 = TripletAttention()
        self.fusion3 = TripletAttention()
        self.fusions = [self.fusion1,self.fusion2,self.fusion3]

        if ife == 'c':
            self.cnn_select = CNN_qulv()
        else:
            self.cnn_select = CNN_Entropy()

    def _upsample_fusion(self, x, y, k):
        fusion = self.fusions[k]
        _, _, H, W = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        out = fusion(y, x)
        return out

    def concatenation(self,p2, p3, p4, p5):
        x0_h, x0_w = p2.size(2), p2.size(3)
        p3 = F.interpolate(p3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)  # 2
        p4 = F.interpolate(p4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)  # 4
        p5 = F.interpolate(p5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)  # 8
        out_cat = torch.cat([p2, p3, p4, p5], dim=1)
        out_cat = self.conv(out_cat)
        return out_cat

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_fusion(p5, self.latlayer1(c4),2)
        p3 = self._upsample_fusion(p4, self.latlayer2(c3),1)
        p2 = self._upsample_fusion(p3, self.latlayer3(c2),0)
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        #IFE
        c2_e = self.cnn_select(c2, 0.5)

        #cat
        out = self.concatenation(p2, p3, p4, p5)
        out = torch.cat([out, c2_e], dim=1)
        return out


class HAFFP2(nn.Module):
    def __init__(self,channels=[64, 128, 256, 512],out_channal=256,ife = 'c'):
        super(HAFFP2, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(channels[3], out_channal, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(out_channal, out_channal, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(out_channal, out_channal, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(out_channal, out_channal, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(channels[2], out_channal, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(channels[1], out_channal, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(channels[0], out_channal, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(in_channels=out_channal * 4, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.fusion1 = GAM_Attention()
        self.fusion2 = GAM_Attention()
        self.fusion3 = GAM_Attention()
        self.fusions = [self.fusion1,self.fusion2,self.fusion3]

        if ife == 'c':
            self.cnn_select = CNN_qulv()


    def _upsample_fusion(self, x, y, k):
        fusion = self.fusions[k]
        _, _, H, W = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        out = fusion(y, x)
        return out

    def concatenation(self,p2, p3, p4, p5):
        x0_h, x0_w = p2.size(2), p2.size(3)
        p3 = F.interpolate(p3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)  # 2
        p4 = F.interpolate(p4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)  # 4
        p5 = F.interpolate(p5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)  # 8
        out_cat = torch.cat([p2, p3, p4, p5], dim=1)
        out_cat = self.conv(out_cat)
        return out_cat

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_fusion(p5, self.latlayer1(c4),2)
        p3 = self._upsample_fusion(p4, self.latlayer2(c3),1)
        p2 = self._upsample_fusion(p3, self.latlayer3(c2),0)
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        #cat
        out = self.concatenation(p2, p3, p4, p5)

        return out
if __name__ == '__main__':
    input0 = torch.randn(8, 512, 8, 8)
    input1 = torch.randn(8, 256, 16, 16)
    input2 = torch.randn(8, 128, 32, 32)
    input3 = torch.randn(8, 64, 64, 64)
    # input = [input3,input2,input1,input0]

    fafm = HAFFP2([64, 128, 256, 512])
    output = fafm(input3,input2,input1,input0)
    print('^&',output.shape)
    from thop import profile

    flops, params = profile(fafm, inputs=(input3,input2,input1,input0))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
