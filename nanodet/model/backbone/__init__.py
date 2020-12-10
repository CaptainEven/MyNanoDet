# encoding=utf-8

import copy
from .resnet import ResNet
from .ghostnet import GhostNet
from .shufflenetv2 import ShuffleNetV2
from .mobilenetv2 import MobileNetV2


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')

    if name == 'ResNet':
        return ResNet(**backbone_cfg)
    elif name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    elif name == 'GhostNet':
        backbone_cfg.pop('model_size')
        backbone_cfg.pop('activation')
        return GhostNet(**backbone_cfg)
    elif name == 'MobileNetV2':
        backbone_cfg.pop('model_size')
        backbone_cfg.pop('activation')
        return MobileNetV2(**backbone_cfg)
    else:
        raise NotImplementedError

