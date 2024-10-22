# @auther:guopeng
# @time:2023/9/25 18:51
# @file:flop_param.py
# @description:
from models.HA2PNet import HA2PNet
from thop import profile
import torch

model = HA2PNet(nclass=6, backbone='effnetv2_s', pretrained_base=True,
                              frozen_stages=-1, norm_eval=False,
                              seg_feat_channels=256, stacked_convs=7,
                              ins_out_channels=256,kernel_size=1,output_stride=4)


randn_input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(randn_input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
