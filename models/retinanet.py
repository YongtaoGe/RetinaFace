from . import backbones as backbones_mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import math
import os

__all__ = ['RetinaNet']
class RetinaNet(nn.Module):

    def __init__(self, phase, cfg):
        super(RetinaNet, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        backbones = 'ResNet18FPN'
        if not isinstance(backbones, list):
            backbones = [backbones]
        self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})

        # classification and box regression heads
        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        # anchors = len(self.ratios) * len(self.scales)
        anchors_per_layer = 3*3
        self.cls_head = make_head(self.num_classes * anchors_per_layer)
        self.box_head = make_head(4 * anchors_per_layer)
        self.ldmk_head = make_head((10 * anchors_per_layer))

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        # if self.phase == 'train':
            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         if m.bias is not None:
            #             nn.init.xavier_normal_(m.weight.data)
            #             m.bias.data.fill_(0.02)
            #         else:
            #             m.weight.data.normal_(0, 0.01)
            #     elif isinstance(m, nn.BatchNorm2d):
            #         m.weight.data.fill_(1)
            #         m.bias.data.zero_()

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

    def forward(self, x):
        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))

        # Heads forward pass
        conf = [self.cls_head(t).permute(0, 2, 3, 1).contiguous() for t in features]
        loc = [self.box_head(t).permute(0, 2, 3, 1).contiguous() for t in features]
        ldmk = [self.ldmk_head(t).permute(0, 2, 3, 1).contiguous() for t in features]

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes)),
                          ldmk.view(ldmk.size(0), -1, 10))

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes),
                      ldmk.view(ldmk.size(0), -1, 10))

        return output