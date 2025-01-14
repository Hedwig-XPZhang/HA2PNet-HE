import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .necks.jpfm import JPFM
from .base_ops.CoordConv import CoordConv2d
from .necks.fpn import FPN
from .necks.afpn import AFPN
from .necks.gd import RepGDNeck,GDNeck

from .base_ops.involution import Involution
from models.necks.hamfa import HAMFA,HAFF
from .backbone.cotnext import CotLayer
from .attention.ife import CNN_qulv,CNN_Entropy


class HA2PNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50',  pretrained_base=True, frozen_stages=-1,norm_eval=False, seg_feat_channels=256, stacked_convs=7,ins_out_channels=256, kernel_size=1,output_stride=4, **kwargs):
        super(HA2PNet, self).__init__(backbone, pretrained_base=pretrained_base, frozen_stages=frozen_stages,norm_eval=norm_eval,  **kwargs)

        if 'res' in backbone:
            self.fpn=FPN()
            self.forward=self.forward_res
        elif 'cot' in backbone:
            self.fpn = FPN()
            self.forward = self.forward_res
        elif 'effnetv2_s' in backbone:

            self.haffp = HAMFA([48, 64, 160, 256],256,'c')
            self.forward = self.forward_ham

        elif 'mobilenetv3_l' in backbone:

            self.haffp = HAMFA([24, 40, 112, 160], 256, 'c')
            self.forward = self.forward_ham

        elif 'fastvitma36' in backbone:
            # c_list = [76, 152, 304, 608]
            self.haffp = HAMFA([76, 152, 304, 608], 256, 'c')
            self.forward = self.forward_ham
        elif 'fastvitsa' in backbone:
            # c_list = [64, 128, 256, 512]
            self.haffp = HAMFA([64, 128, 256, 512], 256, 'c')
            self.forward = self.forward_my
        elif 'hrnet' in backbone:
            self.haffp = HAMFA([64, 128, 256, 512], 256, 'c')
            self.forward = self.forward_my
        elif 'swin' in backbone:
            self.fpn=FPN(channels=[192,384,768,1536])
            self.forward=self.forward_swin
        else:
            raise NotImplementedError
        self.output_stride=output_stride
        self.heads=_Head(num_classes=nclass,
                                 in_channels=280,
                                 seg_feat_channels=seg_feat_channels,
                                 stacked_convs=stacked_convs,
                                 ins_out_channels=ins_out_channels,
                                 kernel_size=kernel_size)

    def forward_swin(self, x):
        c2, c3, c4, c5 = self.pretrained(x)
        c2, c3, c4, c5=self.fpn(c2, c3, c4, c5)
        x0_h, x0_w = c2.size(2), c2.size(3)
        c3 = F.interpolate(c3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c5 = F.interpolate(c5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        cat_x = torch.cat([c2,c3,c4,c5], 1)

        output=self.heads(cat_x, cat_x, cat_x)
        return output

    def forward_res(self, x):
        c1, c2, c3, c4, c5 = self.base_forward(x)
        c2, c3, c4, c5=self.fpn(c2, c3, c4, c5)
        x0_h, x0_w = c2.size(2), c2.size(3)
        c3 = F.interpolate(c3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c5 = F.interpolate(c5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        cat_x = torch.cat([c2,c3,c4,c5], 1)

        output=self.heads(cat_x, cat_x, cat_x)
        return output

    def forward_ham(self, x):
        # c2, c3, c4, c5 = self.base_forward_mobilev3l(x)  ##
        # c2, c3, c4, c5 = self.base_forward_fastvit(x)
        c2, c3, c4, c5 = self.base_forward_effv2s(x) ##s l m
        # c2, c3, c4, c5 = self.pretrained(x) # hrnet

        f = self.haffp(c2, c3, c4, c5)

        if self.output_stride!=4:
            f2=F.interpolate(f, size=(256//self.output_stride, 256//self.output_stride), mode='bilinear', align_corners=True)
            f3=F.interpolate(f, size=(256//self.output_stride, 256//self.output_stride), mode='bilinear', align_corners=True)
            output = self.heads(f,f2,f3)
        else:
            output = self.heads(f, f, f)

        return output

    def forward_gd(self, x):
        # c2, c3, c4, c5 = self.base_forward_mobilev3l(x)  ##
        c2, c3, c4, c5 = self.base_forward_fastvit(x)
        # c2, c3, c4, c5 = self.base_forward_effv2s(x) ##s l m
        c3, c4, c5 = self.gd([c2, c3, c4, c5])

        x0_h, x0_w = c3.size(2), c3.size(3)
        c4 = F.interpolate(c4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c5 = F.interpolate(c5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        cat_x = torch.cat([c3,c4,c5], 1)  #
        xo = F.interpolate(cat_x, size=(64, 64), mode='bilinear', align_corners=True)

        #cat_x = self.at(cat_x)
       # xo = self.at_f(xo)
        output=self.heads(xo, xo, xo)

        return output

    def forward_cot(self, x):
        c1, c2, c3, c4, c5 = self.base_forward(x)
        # c2, c3, c4, c5=self.fpn(c2, c3, c4, c5)
        x0_h, x0_w = c2.size(2), c2.size(3)
        c3 = F.interpolate(c3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c5 = F.interpolate(c5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        cat_x = torch.cat([c2,c3,c4,c5], 1)


        f1 = self.jpfm_1(cat_x)
        f2 = self.jpfm_2(cat_x)
        f3 = self.jpfm_3(cat_x)
        if self.output_stride != 4:
            f2 = F.interpolate(f2, size=(256 // self.output_stride, 256 // self.output_stride), mode='bilinear',
                               align_corners=True)
            f3 = F.interpolate(f3, size=(256 // self.output_stride, 256 // self.output_stride), mode='bilinear',
                               align_corners=True)
        output = self.heads(f1, f2, f3)
        return output

    def forward_hrnet(self, x):
        c2,c3,c4,c5 = self.pretrained(x)
        x0_h, x0_w = c2.size(2), c2.size(3)
        c3 = F.interpolate(c3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        c5 = F.interpolate(c5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        cat_x = torch.cat([c2,c3,c4,c5], 1)

        f1=self.jpfm_1(cat_x)
        f2=self.jpfm_2(cat_x)
        f3=self.jpfm_3(cat_x)
        if self.output_stride!=4:
            f2=F.interpolate(f2, size=(256//self.output_stride, 256//self.output_stride), mode='bilinear', align_corners=True)
            f3=F.interpolate(f3, size=(256//self.output_stride, 256//self.output_stride), mode='bilinear', align_corners=True)
        output=self.heads(f1,f2,f3)

        return output

class _Head(nn.Module):
    def __init__(self,num_classes,
                 in_channels=256*4,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 ins_out_channels=256,
                 kernel_size=1
                 ):

        super(_Head,self).__init__()
        self.num_classes = num_classes
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.seg_feat_channels = seg_feat_channels
        self.seg_out_channels = ins_out_channels
        self.ins_out_channels = ins_out_channels
        self.kernel_out_channels = (self.ins_out_channels * kernel_size * kernel_size)


        self._init_layers()
        self.init_weight()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            conv = CoordConv2d if i ==0 else nn.Conv2d
            self.kernel_convs.append(nn.Sequential(
                conv(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
            ))
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.seg_feat_channels),
                nn.ReLU(True),
            ))


        self.head_kernel = nn.Conv2d(self.seg_feat_channels, self.kernel_out_channels, 1,padding=0)
        self.head_cate = nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.mask_convs.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.seg_feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.seg_feat_channels),
            nn.ReLU(True),
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.seg_feat_channels),
            nn.ReLU(True),
            ))


        self.mask_convs.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.seg_feat_channels),
            nn.ReLU(True),))

        self.mask_convs.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.seg_feat_channels),
            nn.ReLU(True)))

        self.head_mask = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(self.seg_out_channels),
            nn.ReLU(True))

    def init_weight(self):
        prior_prob = 0.01
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.normal_(self.head_cate.weight, std=0.01)
        torch.nn.init.constant_(self.head_cate.bias, bias_init)

    def forward(self, feats,f2,f3):

        # feature branch
        mask_feat=feats
        for i, mask_layer in enumerate(self.mask_convs):
            mask_feat = mask_layer(mask_feat)
        feature_pred = self.head_mask(mask_feat)

        # kernel branch
        kernel_feat=f2
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.head_kernel(kernel_feat)

        # cate branch
        cate_feat=f3
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.head_cate(cate_feat)
        return feature_pred, kernel_pred, cate_pred

