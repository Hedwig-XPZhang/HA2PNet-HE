import torch
from torch import nn
import torch.nn.functional as F





class BCEWithLogitsLossWithOHEM(nn.Module):

    def __init__(self, ohem_ratio=1.0, pos_weight=None, eps=1e-7):
        super(BCEWithLogitsLossWithOHEM, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=pos_weight)
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def set_ohem_ratio(self, ohem_ratio):
        self.ohem_ratio = ohem_ratio

def _ohem_mask(loss, ohem_ratio):
    with torch.no_grad():
        values, _ = torch.topk(loss.reshape(-1),
                               int(loss.nelement() * ohem_ratio))
        mask = loss >= values[-1]
    return mask.float()
class CrossEntropyLossWithOHEM(nn.Module):

    def __init__(self, ohem_ratio=1.0, weight=None, ignore_index=-100,
                 eps=1e-7):
        super(CrossEntropyLossWithOHEM, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_index,
                                             reduction='none')
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def set_ohem_ratio(self, ohem_ratio):
        self.ohem_ratio = ohem_ratio
