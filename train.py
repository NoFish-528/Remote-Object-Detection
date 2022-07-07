from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_original import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# 参数配置
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],  # 数据集
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,                     # 数据集路径
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',         # backbone
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,       # 批量处理大小
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')   # 从检查点状态恢复训练
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,       # 权重衰变量，防止过拟合
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',                # 模型保存路径
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

# 若使用CUDA
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# 实现可视化
if args.visdom:
    import visdom
    viz = visdom.Visdom()

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'], MEANS))
    # 当训练数据集为VOC时
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    # 构建SSD网络模型
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    # for name, paras in net.named_parameters():
    #     print(name,':', paras)

    # 实现GPU并行计算
    if args.cuda:
        #net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        print('cuda is avalible')

    # 从断点处恢复模型
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)

    # 加载模型(vgg16_reducedfc.pth
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    # 不从断点处恢复模型，则重新初始化模型
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        # for i in range(6):
            # ssd_net.attention[i].apply(weights_init)
            # ssd_net.Fusion.attention2[i].apply(weights_init)
            # ssd_net.Fusion.change_channels[i].apply(weights_init)
            # ssd_net.Fusion.global_channels[i].apply(weights_init)
            # ssd_net.Down_Up_Sampling.Change_Channel.apply(weights_init)
            # ssd_net.FBS.apply(weights_init)
        # for j in range(5):
        #     ssd_net.Down_Up_Sampling.Down_sampling[j].apply(weights_init)
        #     ssd_net.Down_Up_Sampling.Up_Sampling[j].apply(weights_init)
        #
        # ssd_net.Down_Up_Sampling.apply(weights_init)
       # ssd_net.vgg.apply(weights_init)

    if args.cuda:
        net.cuda()

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 损失函数
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    # Tensorboard可视化
    write = SummaryWriter('runs/SSD300_HRRSD')
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    # 可视化
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    # DataLoader实现了一个并行读入图像、标签的功能。
    # data_loader获取到的数据为（batch,channels,height,width)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    # 训练迭代次数
    start_time = time.time()
    per_time = 0
    best_loss = 30
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # 更新学习率
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)
        try:
            images, targets = next(batch_iterator)
            # print(str(iteration),images.size())
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        # 使用cuda
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)for ann in targets]

        # 从批量图片及标签中提取每一张图片及其对应标签
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        # 开始计时
        t0 = time.time()
        # write.add_graph(net, images)
        out = net(images)

        # backprob
        optimizer.zero_grad()   # 梯度清零
        loss_l, loss_c = criterion(out, targets)   # 计算位置损失及置信度损失(batch_sizd的縱LOSS/batch_size）
        loss = loss_l + loss_c     # 总损失
        loss.backward()  # 误差反向传播
        optimizer.step()  # 梯度优化
        # print('after')
        # for name, paras in net.named_parameters():
        #     print(name, ':', paras)
        t1 = time.time()
        # loc_loss += loss_l.data[0]
        # conf_loss += loss_c.data[0]

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        write.add_scalar('Train_Loss', loss.item(), iteration)
        write.add_scalar('Location_Loss', loss_l.item(), iteration)
        write.add_scalar('Confidence_Loss', loss_c.item(), iteration)
        per_time +=(t1 - t0)
        #print(iteration, loss.item())
        if iteration % 1 == 0:
            print('Spend time per 1000: %0.4f sec.' % (per_time))
            # print('timer: %.4f sec.' % (t1 - t0))
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            print('iter:{0} ||Loss:{1:0.4f}||  ||Conf_Loss:{2:0.4f}||  ||Loc_Loss:{3:0.4f}|| \n'
                  .format(iteration, loss.item(), loss_c.item(), loss_l.item()))
            per_time = 0

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        # 保存最好的模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            print('the Best initeration Now:'+ repr(iteration) +' '+ 'Loss: %.4f\n' % (loss.item()))
            torch.save(ssd_net.state_dict(), args.save_folder + '' + 'Best_SSD300_HRRSD' + '.pth')
        # 每10000次迭代，保存一次模型
        #if iteration != 0 and iteration % 10000 == 9999:
            #print('Saving state, iter:', iteration)
            #torch.save(ssd_net.state_dict(), 'weights/ssd400_NWPU_exp1' +
                       #repr(iteration) + '.pth')
    write.close()
    end_time = time.time()
    print('Spend Total Time: %0.4f sec.' % (end_time - start_time))
    print('best loss', best_loss)
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' +'SSD300_HRRSD' + '.pth')

# 调节学习率
def adjust_learning_rate(optimizer, gamma, step):
    """
    Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# 可视化相关函数
def create_vis_plot(_xlabel, _ylabel, _title, _legend,):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
