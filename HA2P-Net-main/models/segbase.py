import torchvision
from .backbone.resnetv1b import resnet50_v1s,resnet101_v1s
from .backbone.res2netv1b import res2net50_v1b
from .backbone.resnext import resnext50_32x4d,resnext101_32x8d
from .backbone.seg_hrnet import hrnet_w18_v2,hrnet_w32,hrnet_w44,hrnet_w48,hrnet_w64
from .backbone.resnextdcn import resnext101_32x8d_dcn

from .backbone.cotnext import cotnext101_2x48d
from .backbone.mobilenetv3 import mobilenet_v3_large,mobilenet_v3_small
from .backbone.efficientnet_v2 import efficientnetv2_s,efficientnetv2_l,efficientnetv2_m
from .backbone.fastVit.fastvit import fastvit_ma36,fastvit_sa12,fastvit_sa36,fastvit_sa24

from .attention.cbam import CBAMBlock


from .backbone.swim_transformer import swim_large
from torch.nn.modules.batchnorm import _BatchNorm
__all__ = ['SegBaseModel']
import torch
import torch.nn as nn
import torch.nn.functional as F
class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
        resnest
        resnext
        res2net
        DLA
    """

    def __init__(self, backbone='enc', pretrained_base=False, frozen_stages=-1,norm_eval=False, **kwargs):
        super(SegBaseModel, self).__init__()
        self.norm_eval=norm_eval
        self.frozen_stages=frozen_stages
        if backbone == 'resnext101dcn':
            self.pretrained = resnext101_32x8d_dcn(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'effnetv2_s':
            self.pretrained = efficientnetv2_s(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'effnetv2_m':
            self.pretrained = efficientnetv2_m(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'effnetv2_l':
            self.pretrained = efficientnetv2_l(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'mobilenetv3_l':
            self.pretrained = mobilenet_v3_large(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'fastvitma36':
            self.pretrained = fastvit_ma36(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'fastvitsa36':
            self.pretrained = fastvit_sa36(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'fastvitsa24':
            self.pretrained = fastvit_sa24(pretrained=pretrained_base, **kwargs)
        elif backbone == 'fastvitsa12':
            self.pretrained = fastvit_sa12(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'cotnext101':
            self.pretrained = cotnext101_2x48d(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'hrnet18':
            self.pretrained = hrnet_w18_v2(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone =='hrnet32':
            self.pretrained=hrnet_w32(pretrained=pretrained_base,  **kwargs)
        elif backbone =='hrnet64':
            self.pretrained=hrnet_w64(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'resnext101':
            self.pretrained = resnext101_32x8d(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'res2net50':
            self.pretrained = res2net50_v1b(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnext50':
            self.pretrained = resnext50_32x4d(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'swin':
            self.pretrained = swim_large(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self._train()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def freeze(self):
        self.set_requires_grad(self.pretrained,False)
        self.set_requires_grad([self.pretrained.conv1,self.pretrained.bn1],True)

    def unfreeze(self):
        self.set_requires_grad([self.pretrained],True)

    def _train(self, mode=True):
        super(SegBaseModel, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            print('Freeze backbone BN using running mean and std')
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if hasattr(self.pretrained, 'conv1'):
                self.pretrained.bn1.eval()
                for m in [self.pretrained.conv1, self.pretrained.bn1]:
                    for param in m.parameters():
                        param.requires_grad = False
            if hasattr(self.pretrained, 'conv2'):
                self.pretrained.bn2.eval()
                for m in [self.pretrained.conv2, self.pretrained.bn2]:
                    for param in m.parameters():
                        param.requires_grad = False
            if hasattr(self.pretrained, 'stem'):
                self.pretrained.stem.bn.eval()
                for m in [self.pretrained.stem.conv, self.pretrained.stem.bn]:
                    for param in m.parameters():
                        param.requires_grad = False

        print(f'Freezing backbone stage {self.frozen_stages}')

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.pretrained, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # trick: only train conv1 since not use ImageNet mean and std image norm
        if hasattr(self.pretrained,'stem'):
            print('active train conv and bn')
            self.set_requires_grad([self.pretrained.stem.conv, self.pretrained.stem.bn],True)
        if hasattr(self.pretrained, 'conv1'):
            print('active train conv1 and bn1')
            self.set_requires_grad([self.pretrained.conv1, self.pretrained.bn1], True)

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        c1 = self.pretrained.maxpool(x)
        c2 = self.pretrained.layer1(c1)
        c3 = self.pretrained.layer2(c2)
        c4 = self.pretrained.layer3(c3)
        c5 = self.pretrained.layer4(c4)
        return c1, c2, c3, c4, c5

    def base_forward_effv2s(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.stem(x)

        c2 = self.pretrained.blocks[:6](x)
        # c2 = self.att1(c2)
        c3 = self.pretrained.blocks[6:10](c2)
        # c3 = self.att2(c3)
        c4 = self.pretrained.blocks[10:25](c3)
        # c4 = self.att3(c4)
        c5 = self.pretrained.blocks[25:](c4)
        # c5 = self.att4(c5)

        return c2, c3, c4, c5

    def base_forward_effv2m(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.stem(x)
        c2 = self.pretrained.blocks[:8](x)
        c3 = self.pretrained.blocks[:13](x)
        c4 = self.pretrained.blocks[:34](x)
        c5 = self.pretrained.blocks(x)

        return c2, c3, c4, c5

    def base_forward_effv2l(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.stem(x)
        c2 = self.pretrained.blocks[:11](x)
        c3 = self.pretrained.blocks[11:18](c2)
        c4 = self.pretrained.blocks[18:47](c3)
        c5 = self.pretrained.blocks[47:](c4)
        return c2, c3, c4, c5
    def base_forward_mobilev3l(self, x):
        """forwarding pre-trained network"""
        c2 = self.pretrained.features[:4](x)
        c3 = self.pretrained.features[4:7](c2)
        c4 = self.pretrained.features[7:13](c3)
        c5 = self.pretrained.features[13:16](c4)
        return c2, c3, c4, c5
    def base_forward_fastvit(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.forward_embeddings(x)
        atts = [self.att1,self.att2,self.att3,self.att4]
        #bns = [self.bn1, self.bn2, self.bn3, self.bn4]
        outs = []
        for idx, block in enumerate(self.pretrained.network):
            x = block(x)
            if idx in [0, 2, 4, 7]:
                # bn = bns.pop(0)
                # xo = bn(x)
                att = atts.pop(0)
                x = att(x)
                outs.append(x)
        return outs
