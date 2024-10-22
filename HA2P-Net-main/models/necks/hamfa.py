import torch
from torch import nn
from models.base_ops.involution import Involution
from torch.nn import init
import torch.nn.functional as F
from models.attention.ife import CNN_qulv,CNN_Entropy


################
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


class HAFF(nn.Module):
    def __init__(self, no_spatial=False):
        super(HAFF, self).__init__()
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

####################
class HAMFA(nn.Module):
    def __init__(self,channels=[48, 64, 160, 256],out_channal=256,ife = 'c'):
        super(HAMFA, self).__init__()
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

        self.fusion1 = HAFF()
        self.fusion2 = HAFF()
        self.fusion3 = HAFF()
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
        c2_e = self.cnn_select(c2, 0.5) #24

        #cat
        out = self.concatenation(p2, p3, p4, p5) #256
        out = torch.cat([out, c2_e], dim=1) #280
        return out



if __name__ == '__main__':
    input0 = torch.randn(8, 512, 8, 8)
    input1 = torch.randn(8, 256, 16, 16)
    input2 = torch.randn(8, 128, 32, 32)
    input3 = torch.randn(8, 64, 64, 64)
    # input = [input3,input2,input1,input0]

    fafm = HAMFA([64, 128, 256, 512])
    output = fafm(input3,input2,input1,input0)
    print('^&',output.shape)
    from thop import profile

    flops, params = profile(fafm, inputs=(input3,input2,input1,input0))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
