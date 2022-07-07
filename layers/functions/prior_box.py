from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
'''
每个特征图先验框设置
'''
# PriorBox返回得是所有先验框得四个参数归一化后的值，即得到所有先验框的位置
class PriorBox(object):
    """
    Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']    # min_dim=300
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):   # 'feature_maps': [38, 19, 10, 5, 3, 1],
            # product(range(f), repeat=2)相当于得到特定特征图的每个像素点
            for i, j in product(range(f), repeat=2):  # product(A,B)=>生成列表A,B元素的笛卡尔积   product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
                f_k = self.image_size / self.steps[k]  # fk为特征图的大小
                # unit center x,y
                cx = (j + 0.5) / f_k   # 每个单元对应先验框的中心点分布在各个单元的中心
                cy = (i + 0.5) / f_k

                # 计算ar=1的先验框尺寸
                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size  # sk为先验框尺寸相对于原始Image的比例
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        # 归一化，将output限制在[0,1]之间
        if self.clip:
            output.clamp_(max=1, min=0)
        # output存储每个单元对应的每个先验框的[cx,cy,w,h]
        return output
