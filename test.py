from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd_fusion import build_ssd
from tensorboardX import SummaryWriter
from layers.modules import MultiBoxLoss
# 配置参数
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
# parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',  # 使用的训练模型
parser.add_argument('--trained_model', default='weights/ Best_SSD400_init_GN_120000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='test/', type=str,     # 结果保存路径
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')   # 置信度阈值
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')   # 是否使用cuda
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')    # 数据加载路径
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

# 若支持cuda
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
# 使用cpu
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# 定义测试网络
def test_net(save_folder, net, cuda, testset, transform,thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'Best_SSD400_init_GN_120000.txt'
    # print(filename)
    num_images = len(testset)    # 测试集大小
    # 开始测试
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        # 前向传播
        y = net(x)      # forward pass
        # 预测
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, 'a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1

# 测试
def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background   # 类别+北京
    net = build_ssd('test', 300, num_classes) # initialize SSD   # 构建SSD结构
    net.load_state_dict(torch.load(args.trained_model))   # 加载训练好的模型
    net.eval()  # 模型评估
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2012', 'test')], None, VOCAnnotationTransform())  # 加载测试数据
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    # 结果检测
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (99, 106, 95)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
