import torch
import torch.nn as nn
import torch.nn.functional as F


"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


""" Deep Supervision Loss"""


def DeepSupervisionLoss(pred, gt):
    d0, d1, d2, d3, d4 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion(d1, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gt)

    return loss0 + loss1 + loss2 + loss3 + loss4


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def TotalStructureLoss(pred, gt):
    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = pred
    loss5 = structure_loss(lateral_map_5, gt)
    loss4 = structure_loss(lateral_map_4, gt)
    loss3 = structure_loss(lateral_map_3, gt)
    loss2 = structure_loss(lateral_map_2, gt)
    return loss2 + loss3 + loss4 + loss5


def criterion(loss_type):
    if loss_type == 'DeepSupervisionLoss':
        return DeepSupervisionLoss
    elif loss_type == 'BCELoss':
        return BCELoss()
    elif loss_type == 'DiceLoss':
        return DiceLoss()
    elif loss_type == 'BceDiceLoss':
        return BceDiceLoss()
    elif loss_type == 'TotalStructureLoss':
        return TotalStructureLoss
    else:
        raise NotImplementedError


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

