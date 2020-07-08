
import torch.nn as nn
from ..duc.DUC import DUC
from .mobilenet import MobileNetV2
from config.config import pose_cls, DUC_idx
from config.model_cfg import DUC_cfg, mobile_opt

n_classes = pose_cls
DUCs = DUC_cfg[DUC_idx]


def createModel(cfg=None):
    return MobilePose(cfg)


class MobilePose(nn.Module):
    def __init__(self, cfg):
        setting = mobile_opt[cfg]
        super(MobilePose, self).__init__()
        # print(setting)
        self.mobile = MobileNetV2(inverted_residual_setting=setting)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(320, DUCs[0], upscale_factor=2)
        self.duc2 = DUC(int(DUCs[0]/4), DUCs[1], upscale_factor=2)
        #self.duc3 = DUC(128, 256, upscale_factor=2)
        #self.duc4 = DUC(256, 512, upscale_factor=2)
        self.conv_out = nn.Conv2d(
            int(DUCs[1]/4), n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.mobile(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        #out = self.duc3(out)
        #out = self.duc4(out)

        out = self.conv_out(out)
        return out
