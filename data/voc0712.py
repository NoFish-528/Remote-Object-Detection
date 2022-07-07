"""
VOC Dataset Classes
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot
"""
'''
VOC数据集相关定义及处理
'''
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
'''
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
'''
#
VOC_CLASSES = (  # always index 0
    'airplane', 'ship', 'storage tank', 'baseball diamond',
    'tennis court', 'basketball court', 'ground track field',
    'habor', 'bridge', 'vehicle', 'crossroad', 'T junction', 'parking lot')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "VOCNWPU/")

# Voc数据集转换(格式为标签与索引对应且bbox坐标的tensor)
class VOCAnnotationTransform(object):
    """
    Transforms a VOC annotation into a Tensor of bbox coords and label index Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


# 类VOCDetetction继承于torch.utils.data.Dataset类别，需要定义imgs,getitem、__len__方法。
# python的__getitem__方法可以让对象实现迭代功能。在这里，会返回单张图像及其标签
class VOCDetection(data.Dataset):
    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the target `annotation` (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load (default: 'VOC2007')
    """

    def __init__(self, root,
                 # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 image_sets=[('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root                            # 设置数据集的根目录
        # print(self.root)
        self.image_set = image_sets                 # 设置要选用的数据集
        self.transform = transform                  # 定义图像转换方法
        self.target_transform = target_transform    # 定义标签的转换方法
        self.name = dataset_name                    # 定义数据集名称
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')    # 记录标签的位置
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')      # 记录图像的位置
        self.ids = list()       # 记录数据集中的所有图像的名字
        # 读入数据集中的图像名称，可以依照该名称和_annopath、_imgpath推断出图片、描述文件存储的位置
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            # print(rootpath)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]      # 获取index对应的img名称
        #print(img_id)
        target = ET.parse(self._annopath % img_id).getroot()        # 读取xml文件
        #print(target)
        img = cv2.imread(self._imgpath % img_id)      # 获取图像
        #print(self._imgpath)
        #print(img)
        height, width, channels = img.shape         # 获取图像的尺寸

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)       # 获取target
# 数据增强
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])       # 对图像、target进行转换
            # to rgb
            img = img[:, :, (2, 1, 0)]          # opencv读入图像的顺序是BGR，该操作将图像转为RGB
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # 返回image、label、宽高.这里的permute(2,0,1)是将原有的三维（28，28，3）变为（3，28，28），将通道数提前，为了统一torch的后续训练操作。
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        """
        Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        """
        Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        """
        Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
