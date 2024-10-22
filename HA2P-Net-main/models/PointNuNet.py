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
from aff_net.fusion import AFF,iAFF
from .base_ops.involution import Involution
from .attention.myAttention import DAFM,TripletAttention,GAM_Attention,HAFFP,HAFFP2
from .backbone.cotnext import CotLayer
from .attention.ife import CNN_qulv,CNN_Entropy


class PointNuNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50',  pretrained_base=True, frozen_stages=-1,norm_eval=False, seg_feat_channels=256, stacked_convs=7,ins_out_channels=256, kernel_size=1,output_stride=4, **kwargs):
        super(PointNuNet, self).__init__(backbone, pretrained_base=pretrained_base, frozen_stages=frozen_stages,norm_eval=norm_eval,  **kwargs)

        if 'res' in backbone:
            self.fpn=FPN()
            self.forward=self.forward_res
        elif 'cot' in backbone:
            self.fpn = FPN()
            self.forward = self.forward_res
        elif 'effnetv2_s' in backbone:
            # self.afpn = AFPN([48, 64, 160, 256])
            # self.forward = self.forward_eff
            # c_list = [48, 64, 160, 256]
            # self.fpn = FPN(channels=[48, 64, 160, 256])
            # self.jpfm = JPFM(in_channel=528)
            # self.forward=self.forward_eff

            self.haffp = HAFFP([48, 64, 160, 256],256,'c')
            self.forward = self.forward_my
            # self.gd = RepGDNeck(channels_list=[24, 48, 64, 160, 256, 64, 48, 64, 160],
            #                     num_repeats=4,
            #
            #                     extra_cfg=dict(
            #                                     norm_cfg=dict(type='BN', requires_grad=True),
            #                                     depths=2,
            #                                     fusion_in=528,  ##c2345
            #                                     fusion_act=dict(type='ReLU'),
            #                                     fuse_block_num=3,
            #                                     embed_dim_p=112,  ##
            #                                     embed_dim_n=368,  ### p3+p4+c5
            #                                     key_dim=8,
            #                                     num_heads=4,
            #                                     mlp_ratios=1,
            #                                     attn_ratios=2,
            #                                     c2t_stride=2,
            #                                     drop_path_rate=0.1,
            #                                     trans_channels=[64, 48, 64, 160],  ###
            #                                     pool_mode='torch'
            #                                     )
            #                     )
            # self.forward = self.forward_gd
        elif 'effnetv2_m' in backbone:
            self.afpn = AFPN(in_channels=[48,80,176,512])
            #self.forward = self.forward_eff

        elif 'effnetv2_l' in backbone:

            self.haffp = HAFFP([64, 96, 224, 640], 256, 'c')
            self.forward = self.forward_my
            # self.afpn = AFPN(in_channels=[64,96,224,640])
            # self.forward = self.forward_eff
            # self.gd = RepGDNeck(channels_list=[32, 64, 96, 224, 640, 96, 64, 96, 224],
            #                     num_repeats=10,##
            #                     extra_cfg=dict(
            #                         norm_cfg=dict(type='BN', requires_grad=True),
            #                         depths=2,
            #                         fusion_in=1024,  ##c2345
            #                         fusion_act=dict(type='ReLU'),
            #                         fuse_block_num=3,
            #                         embed_dim_p=192,  ##
            #                         embed_dim_n=800,  ### p3+p4+c5
            #                         key_dim=8,
            #                         num_heads=8,
            #                         mlp_ratios=1,
            #                         attn_ratios=2,
            #                         c2t_stride=2,
            #                         drop_path_rate=0.1,
            #                         trans_channels=[96, 64, 96, 224],  ###
            #                         pool_mode='torch'
            #                     )
            #                 )
            # self.forward = self.forward_gd
        elif 'mobilenetv3_l' in backbone:
            # self.fusion = AFF(channels=256)
            self.haffp = HAFFP([24, 40, 112, 160], 256, 'c')
            self.forward = self.forward_my

            # self.gd = RepGDNeck(channels_list=[16, 24, 40, 112, 160, 40, 24, 40, 112],
            #                     num_repeats=4,
            #                     extra_cfg=dict(
            #                         norm_cfg=dict(type='BN', requires_grad=True),
            #                         depths=2,
            #                         fusion_in=336,  ##c2345
            #                         fusion_act=dict(type='ReLU'),
            #                         fuse_block_num=3,
            #                         embed_dim_p=64,  ##
            #                         embed_dim_n=224,  ### p3+p4+c5
            #                         key_dim=8,
            #                         num_heads=4,
            #                         mlp_ratios=1,
            #                         attn_ratios=2,
            #                         c2t_stride=2,
            #                         drop_path_rate=0.1,
            #                         trans_channels=[40, 24, 40, 112],  ###
            #                         pool_mode='torch'
            #                     )
            #                     )
            # self.forward = self.forward_gd
        elif 'fastvitma36' in backbone:

            c_list = [76, 152, 304, 608]
            # self.fusion = DAFM(256)
            # self.fusion = GAM_Attention(256,256,4) #DAFM
            self.cnn_select = CNN_Entropy()
            self.ratio = [0.75, 0.5, 0.5, 0.25]
            # self.cnn_select = CNN_qulv()
            self.fusion = TripletAttention()
            self.fusion1 = TripletAttention()
            self.fusion2 = TripletAttention()
            self.expand3 = nn.Conv2d(in_channels=c_list[3], out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.expand2 = nn.Conv2d(in_channels=c_list[2], out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.expand1 = nn.Conv2d(in_channels=int(c_list[1] * (1 + self.ratio[1])), out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.expand0 = nn.Conv2d(in_channels=int(c_list[0] * (1 + self.ratio[0])), out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.conv = nn.Conv2d(in_channels=256 * 4, out_channels=256, kernel_size=1, stride=1, padding=0)
            self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.forward = self.forward_my
        elif 'fastvitsa' in backbone:
            c_list = [64, 128, 256, 512]
            # self.fusion = DAFM(256)
            # self.fusion = GAM_Attention(256,256,4) #DAFM
            # self.cnn_select = CNN_Entropy()
            # self.ratio = [0.75,0.5,0.5,0.25]
            # self.cnn_select = CNN_qulv()
            # self.cfm = CFM(1024)
            self.fusion = TripletAttention()
            self.fusion1 = TripletAttention()
            self.fusion2 = TripletAttention()
            self.expand3 = nn.Conv2d(in_channels=c_list[3], out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.expand2 = nn.Conv2d(in_channels=c_list[2], out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.expand1 = nn.Conv2d(in_channels=c_list[1], out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.expand0 = nn.Conv2d(in_channels=c_list[0], out_channels=256, kernel_size=1,
                                     stride=1, padding=0)
            self.conv = nn.Conv2d(in_channels=256 * 4, out_channels=256, kernel_size=1, stride=1, padding=0)
            self.smooth1 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
            self.smooth2 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
            self.smooth3 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
            self.forward = self.forward_my
            # self.gd = RepGDNeck(channels_list=[16, 64, 128, 256, 512, 128, 64, 128, 256],
            #                  num_repeats=4,
            #                  extra_cfg=dict(
            #                                 norm_cfg=dict(type='BN', requires_grad=True),
            #                                 depths=2,
            #                                 fusion_in=960,  ##c2345
            #                                 fusion_act=dict(type='ReLU'),
            #                                 fuse_block_num=3,
            #                                 embed_dim_p=128,  ##
            #                                 embed_dim_n=704,  ### p3+p4+c5
            #                                 key_dim=8,
            #                                 num_heads=4,
            #                                 mlp_ratios=1,
            #                                 attn_ratios=2,
            #                                 c2t_stride=2,
            #                                 drop_path_rate=0.1,
            #                                 trans_channels=[128, 64, 128, 256],  ###
            #                                 pool_mode='torch'
            #                                 )
            #                  )
            # self.forward = self.forward_gd
        elif 'hrnet' in backbone:
            # if '32' in backbone:
            #     c=480
            # elif '64' in backbone:
            #     c=960
            # elif '18' in backbone:
            #     c=270
            # else:
            #     raise NotImplementedError
            # self.jpfm_1=JPFM(in_channel=c)
            # self.jpfm_2=JPFM(in_channel=c)
            # self.jpfm_3=JPFM(in_channel=c)
            # self.forward=self.forward_hrnet
            self.haffp = HAFFP2([64, 128, 256, 512], 256, 'e')
            self.forward = self.forward_my
        elif 'swin' in backbone:
            self.fpn=FPN(channels=[192,384,768,1536])
            self.forward=self.forward_swin
        else:
            raise NotImplementedError
        self.output_stride=output_stride
        self.heads=_PointNuNetHead(num_classes=nclass,
                                 in_channels=256,  #gd_effv2s-272   l-384  m-304    ##gd mobilev3 l 176   ##gd fastvitma36 532  sa12 448
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

    def forward_eff(self, x):
        c2, c3, c4, c5 = self.base_forward_effv2s(x)  ###
        # c2, c3, c4, c5=self.afpn([c2, c3, c4, c5])

        # x0_h, x0_w = c2.size(2), c2.size(3)
        # c3 = F.interpolate(c3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # c4 = F.interpolate(c4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # c5 = F.interpolate(c5, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # cat_x = torch.cat([c2,c3,c4,c5], 1)
        # f1 = self.jpfm(cat_x)

        f1 = self.fpn(c2, c3, c4, c5)

        output = self.heads(f1, f1, f1)

        return output

    def forward_my(self, x):
        # c2, c3, c4, c5 = self.base_forward_mobilev3l(x)  ##
        # c2, c3, c4, c5 = self.base_forward_fastvit(x)
        # c2, c3, c4, c5 = self.base_forward_effv2s(x) ##s l m
        c2, c3, c4, c5 = self.pretrained(x) # hrnet

        f = self.haffp(c2, c3, c4, c5)

        if self.output_stride!=4:
            f2=F.interpolate(f, size=(256//self.output_stride, 256//self.output_stride), mode='bilinear', align_corners=True)
            f3=F.interpolate(f, size=(256//self.output_stride, 256//self.output_stride), mode='bilinear', align_corners=True)
            output = self.heads(f,f2,f3)
        else:
            output = self.heads(f, f, f)
        #######################AFF-fpn#######################################
        # c2_e = self.cnn_select(c2, 0.5)

        #
        # p5 = self.expand3(c5)
        # c4 = self.expand2(c4)
        # c3 = self.expand1(c3)
        # c2 = self.expand0(c2)
        # # c5 = self.dconv1(p5) #16  up
        # c5 = F.interpolate(p5, size=(16, 16), mode='bilinear', align_corners=True)
        # p4 = self.fusion(c4,c5)
        # # c4 = self.dconv2(p4) #32
        # c4 = F.interpolate(p4, size=(32, 32), mode='bilinear', align_corners=True)
        # p3 = self.fusion1(c3,c4)
        # # c3 = self.dconv3(p3) #64
        # c3 = F.interpolate(p3, size=(64, 64), mode='bilinear', align_corners=True)
        # p2 = self.fusion2(c2, c3) #256*64*64
        #
        # p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)
        #
        # x0_h, x0_w = p2.size(2), p2.size(3)
        # p3 = F.interpolate(p3, size=(x0_h, x0_w), mode='bilinear', align_corners=True) #2
        # p4 = F.interpolate(p4, size=(x0_h, x0_w), mode='bilinear', align_corners=True) #4
        # p5 = F.interpolate(p5, size=(x0_h, x0_w), mode='bilinear', align_corners=True) #8
        #
        # cat_x = torch.cat([p2,p3, p4, p5], 1)
        # # cat_x = self.cfm(cat_x)
        # cat_x = self.conv(cat_x)
        # cat_x = torch.cat([cat_x, c2_e], dim=1)
        # output = self.heads(cat_x, cat_x, cat_x)
        ############################################################################
        return output

    def forward_gd(self, x):
        # c2, c3, c4, c5 = self.base_forward_mobilev3l(x)  ##
        c2, c3, c4, c5 = self.base_forward_fastvit(x)
        # c2, c3, c4, c5 = self.base_forward_effv2s(x) ##s l m
        c3, c4, c5 = self.gd([c2, c3, c4, c5])

        x0_h, x0_w = c3.size(2), c3.size(3)
        # c3 = F.interpolate(c3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
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
        # output=self.heads(cat_x, cat_x, cat_x)
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

class _PointNuNetHead(nn.Module):
    def __init__(self,num_classes,
                 in_channels=256*4,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 ins_out_channels=256,
                 kernel_size=1
                 ):

        super(_PointNuNetHead,self).__init__()
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
            # nn.ConvTranspose2d(self.seg_feat_channels, self.seg_feat_channels, 4, 2, padding=1, output_padding=0,bias=False),
            # nn.BatchNorm2d(self.seg_feat_channels),
            # nn.ReLU(True),
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.seg_feat_channels),
            nn.ReLU(True),))

        self.mask_convs.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.ConvTranspose2d(self.seg_feat_channels, self.seg_feat_channels, 4, 2, padding=1, output_padding=0,bias=False),
            # nn.BatchNorm2d(self.seg_feat_channels),
            # nn.ReLU(True),
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


class ExBlock(nn.Module):


    def __init__(self, in_channels, out_channels, stride=1):
        super(ExBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Involution(kernel_size=3, channels=out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1,stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        # residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        residual = out
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out
class Deconv(nn.Module):


    def __init__(self, channels):
        super(Deconv, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = Involution(kernel_size=3, channels=channels, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1,stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        # residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        residual = out
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out

