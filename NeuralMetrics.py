from math import exp
import torch
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def evaluate_jaccard(result:torch.Tensor, target : torch.Tensor)->torch.Tensor :
    result = result.detach()
    target = target.detach()
    axis = (2, 3)
    intersection = torch.sum(torch.abs(target * result), axis)
    smooth = 1e-9
    union = torch.sum(target, axis) + torch.sum(result, axis) - intersection
    return (intersection + smooth) / (union + smooth)


class IntersectionOverUnion(torch.nn.Module):
    def __init__(self):
        super(IntersectionOverUnion, self).__init__()

    def forward(self, result, target):
        result = result[:, 1:].detach()
        target = target[:, 1:].detach()
        axis = (1, 2, 3)
        intersection = torch.sum(torch.abs(target * result), axis)
        smooth = 1e-9
        union = torch.sum(target, axis) + torch.sum(result, axis) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()


class SpatialMetric(torch.nn.Module):
    def __init__(self):
        super(SpatialMetric, self).__init__()

    def forward(self, result, target):
        return 1.0 - torch.nn.functional.l1_loss(result, target)

'''
https://en.wikipedia.org/wiki/Structural_similarity
'''

class SSIM(torch.nn.Module):
    def __init__(self, dimension):
        super(SSIM, self).__init__()
        self.conv2d = torch.nn.Conv2d(dimension, dimension, kernel_size=11, padding=11//2, groups=dimension, bias= False)
        self.conv2d.weight.data=create_window(11, dimension)
        self.C1 = float(0.01 ** 2)
        self.C2 = float(0.03 ** 2)

    def forward(self, img1, img2):
        mu1 =  self.conv2d(img1)
        mu2 =  self.conv2d(img2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = self.conv2d(img1 * img1) - mu1_sq
        sigma2_sq = self.conv2d(img2 * img2) - mu2_sq
        sigma12 = self.conv2d(img1 * img2) - mu1_mu2
        cs_map = (2 * sigma12 + self.C2) / (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = ((2 * mu1_mu2 + self.C1) / (mu1_sq + mu2_sq + self.C1)) * cs_map
        return ssim_map.mean()
