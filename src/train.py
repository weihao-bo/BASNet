"""
ModelName: train
Description: 
Author：bwh
Date：2022/2/5 15:40
"""
import sys

sys.path.insert(0, '../')  # 这样新添加的目录会优先于其他目录被import检查，程序退出后失效
sys.dont_write_bytecode = True  # 设置Python解释器不生成字节码pyc文件

import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset as dataset
from utils import adaptive_pixel_intensity_loss, structure_loss, iou_loss, cel_loss
from net.BASNet import BASNet
from apex import amp  # 英伟达提供了混合精度训练工具Apex。号称能够在不降低性能的情况下，将模型训练的速度提升2-4倍，训练显存消耗减少为之前的一半


def validate(model, val_loader, nums):  # 进入验证模式，计算验证集的mae
    """
    model:使用的网络
    val_loader：打包的验证集数据
    nums：验证集数量
    """
    model.train(False)
    avg_mae = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()  # 将图片和标注转成cuda格式
            out = model(image)  # 接收模型预测的结果
            pred = torch.sigmoid(out[0])
            avg_mae += torch.abs(pred - mask[0]).mean()  # 计算评价mae

    model.train(True)

    return (avg_mae / nums).item()


def train(Dataset, Network):  # 传入自定义的dataset与网络
    # 设置随机种子保证结果可以复现
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.set_device(1)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 配置训练集与验证集
    cfg = Dataset.Config(datapath='../data/CHINA-FIRE', savepath='../out/BASNet/CHINA', mode='train', batch=3, lr=0.001,
                         momen=0.9, decay=5e-4, epoch=60)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=0)
    val_cfg = Dataset.Config(datapath='../data/CHINA-FIRE', mode='test')
    val_data = Dataset.Data(val_cfg)
    val_num = len(val_data)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    min_mae = 1.0  # 初始化最小mae
    best_epoch = 0  # 初始化最好的epoch

    # 通过传入的网络定义模型
    net = Network()
    net.train(True)

    net.cuda()

    # 保存参数列表
    enc_params, dec_params = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            enc_params.append(param)
        else:
            dec_params.append(param)

    # 设置优化器
    optimizer = torch.optim.Adam([{'params': enc_params}, {'params': dec_params}], lr=cfg.lr,
                                 weight_decay=cfg.decay)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')  # 使用半精度进行训练

    for epoch in range(cfg.epoch):
        for step, (image, mask, edge) in enumerate(loader):
            image, mask, edge = image.cuda().float(), mask.cuda().float(), edge.cuda().float()

            """
            更换不同网络要重点修改这一块
            一般是单输出，同时要采用原网络的损失函数
            可以将原网络损失函数放入工具类，然后直接调用
            """
            # BASNet
            out1, out2, out3 = net(image)  # 将打包好的图片输入进网络
            loss1 = adaptive_pixel_intensity_loss(out1, mask)
            loss2 = adaptive_pixel_intensity_loss(out2, mask)
            loss3 = adaptive_pixel_intensity_loss(out3, mask)
            loss = loss1 + loss2 + loss3

            # 进行方向传播更新梯度，如果梯度爆炸自行调整
            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scale_loss:
                loss.backward()
            optimizer.step()

            print('%s   [%d/%d]   loss=%.6f' % (str(datetime.datetime.now())[:-7], epoch + 1, cfg.epoch, loss.item()))

        mae = validate(net, val_loader, val_num)
        print('当前epoch MAE:%s' % mae)
        if mae < min_mae:
            min_mae = mae
            best_epoch = epoch + 1
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
        print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))


if __name__ == '__main__':
    train(dataset, BASNet)
