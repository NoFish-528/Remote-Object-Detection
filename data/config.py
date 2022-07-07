# config.py
import os.path
# gets home dir cross platform
# HOME = os.path.expanduser("~")
'''
VOC类信息， COCO类信息
'''
HOME ='/home/niuzhikang/www'
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (123, 116, 103)

# SSD300 CONFIGS
voc = {
    # 'num_classes': 14,
    'num_classes': 11,
    #'lr_steps': (66663, 133326, 199989),
    'lr_steps': (13520, 27040),
    'max_iter': 30000,
    # 300
    'feature_maps': [38, 19, 10, 5, 3, 1],     # 特征图尺寸
    #'feature_maps': [50, 25, 13, 7, 5, 3],  # 特征图尺寸
    'min_dim': 300,   # 输入图片尺寸要求
    # 'steps': [8, 16, 30, 57, 80, 133],  # 本层卷积输出特征图与输入特征图的点的映射关系
    # 'min_sizes': [35, 65, 116, 167, 218, 269],  # 特征图对应先验框的最小尺寸
    # 'max_sizes': [65, 116, 167, 218, 269, 320],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    #'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],  # 与纵横比有关
    #'variance': [0.1, 0.2],         # 通过超参数variance来调整检测值，在进行边框编码or解码时，对边框的预测值L的四个值进行缩放
    # 400
    #'feature_maps': [50, 25, 13, 7, 5, 3],     # 特征图尺寸
    #'min_dim': 400,   # 输入图片尺寸要求
    'steps': [6, 16, 32, 64, 100, 300],   # 本层卷积输出特征图与输入特征图的点的映射关系
    #'min_sizes': [30, 80, 148, 216, 284, 352],  # 特征图对应先验框的最小尺寸
    #'max_sizes': [80, 148, 216, 284, 352, 420],
    #'min_sizes': [30, 60, 111, 162, 213, 264],  # 特征图对应先验框的最小尺寸
    #'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # 与纵横比有关
    'variance': [0.1, 0.2],         # 通过超参数variance来调整检测值，在进行边框编码or解码时，
    'clip': True,
    'name': 'VOC',
}

# COCO CONFIGS
coco = {
    # 'num_classes': 2,
    'num_classes': 11,
    'lr_steps': (280000, 360000, 400000),
    'max_iter':5000,
    'feature_maps': [50, 25, 13, 7, 5, 3],
    'min_dim': 300,
    'steps': [6, 12, 23, 42, 60, 100],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
