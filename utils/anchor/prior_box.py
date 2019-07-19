import torch
from itertools import product as product
import numpy as np
from math import sqrt as sqrt
from math import ceil

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        # self.variance = cfg['variance']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.dense_anchor = cfg['dense_anchor']
        if phase == 'train':
            self.image_size = (cfg['min_dim'], cfg['min_dim'])
        elif phase == 'test':
            self.image_size = image_size

        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for kk, min_size in enumerate(min_sizes):
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if k == 0 and kk == 0 and self.dense_anchor is True:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            mean += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        mean += [cx, cy, s_kx, s_ky]
                        
                        for ar in self.aspect_ratios[k]:
                            mean += [cx, cy, s_kx/sqrt(ar), s_ky*sqrt(ar)]
                            
#                     import pdb
#                     pdb.set_trace()
#                     print(len(mean))

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output