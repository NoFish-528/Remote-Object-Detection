import os
import sys
module_path = os.path.abspath(os.path.join('..'))
#print(module_path)
#print(sys.path)
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from ssd_original import build_ssd
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
import argparse

parser = argparse.ArgumentParser(description= 'SSD Test')
parser.add_argument('--trained_model', default='../weights/Best_SSD300_NWPU_origin.pth',
                    type=str, help='Trained state_dict file path to open')
args = parser.parse_args()


net = build_ssd('test', 300, 11)    # initialize SSD
# net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
# net.load_state_dict(torch.load('../weights/VOC.pth'))   # 加载训练好的模型
# net.load_weights('../weights/SSD.pth')
net.load_state_dict(torch.load(args.trained_model), strict=False)

# image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
# %matplotlib inline

# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
# 加载测试数据集
testset = VOCDetection(VOC_ROOT, [('2012', 'val')], None, VOCAnnotationTransform())
# val.txt中第60张图片
img_id = 110
image = testset.pull_image(img_id)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
# 显示测试图片
# plt.figure(figsize=(10,10))
# plt.imshow(rgb_image)
# plt.show()

x = cv2.resize(image, (300, 300)).astype(np.float32)  # 将图片尺寸改为300*300
x -= (86.0, 91.0, 82.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
# plt.imshow(x)
# plt.show()
x = torch.from_numpy(x).permute(2, 0, 1)

xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
# y, _ = net(xx)
y= net(xx)
from data import VOC_CLASSES as labels
top_k=10

plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
# plt.show()
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
# print(detections.size(1))
for i in range(detections.size(1)):
    j = 0

    while detections[0,i,j,0] >= 0.6:
        print(detections[0, i, j, 0])
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        #print(coords)
        color = colors[i]
        #print(colors[i])
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        #plt.show()
        j+=1
plt.show()