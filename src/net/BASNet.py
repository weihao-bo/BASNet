"""
ModelName: BASNet
Description: 
Author：bwh
Date：2022/2/5 15:41
"""
from torchaudio.models.conv_tasnet import ConvBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


def RCF(F1, F2):
    return F1 * F2 + F2


class ChannelRecalibration(nn.Module):
    def __init__(self, in_channels):
        super(ChannelRecalibration, self).__init__()
        inter_channels = in_channels // 4  # channel squeezing
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(nn.Linear(in_channels, inter_channels, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(inter_channels, in_channels, bias=False))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(nn.Linear(in_channels, inter_channels, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(inter_channels, in_channels, bias=False))

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = self.avg_fc(self.avg_pool(ftr).squeeze(-1).squeeze(-1))  # [B, C]
        ftr_max = self.max_fc(self.max_pool(ftr).squeeze(-1).squeeze(-1))  # [B, C]
        weights = F.sigmoid(ftr_avg + ftr_max).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        out = weights * ftr
        return out

    def initialize(self):
        weight_init(self)


class GFA(nn.Module):
    # Global Feature Aggregation
    def __init__(self, in_channels, squeeze_ratio=4):
        super(GFA, self).__init__()
        inter_channels = in_channels // squeeze_ratio  # reduce computation load
        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor([0.1]))  # initiate as 0.1
        self.cr = ChannelRecalibration(in_channels)

    def forward(self, ftr):
        B, C, H, W = ftr.size()
        P = H * W
        ftr_q = self.conv_q(ftr).view(B, -1, P).permute(0, 2, 1)  # [B, P, C']
        ftr_k = self.conv_k(ftr).view(B, -1, P)  # [B, C', P]
        ftr_v = self.conv_v(ftr).view(B, -1, P)  # [B, C, P]
        weights = F.softmax(torch.bmm(ftr_q, ftr_k), dim=1)  # column-wise softmax, [B, P, P]
        G = torch.bmm(ftr_v, weights).view(B, C, H, W)
        out = self.delta * G + ftr
        out_cr = self.cr(out)
        return out_cr

    def initialize(self):
        weight_init(self)


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x

    def initialize(self):
        weight_init(self)


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x

    def initialize(self):
        weight_init(self)


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x

    def initialize(self):
        weight_init(self)


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.Sigmoid, ConvBlock)):
            pass
        else:
            m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1,
            bias=False):  # 定义一个3x3的卷积核 步长1 padding1 孔洞卷积1 控制卷积核之间的间距 是否将一个 学习到的 bias 增加输出中，默认是 True 。【可选】
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):  # 定义一个1x1的卷积核
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out + x, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        ch = [64, 64, 128, 256, 512]
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # x[2,3,672,672]
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)  # out1[2,64,128,128]
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)  # out3[2,128,84,84]
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet18-5c106cde.pth'), strict=False)


# 卷积模块
class CBM(nn.Module):
    def __init__(self, channel):
        super(CBM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.conv_3 = conv3x3(channel, channel)
        self.bn_3 = nn.BatchNorm2d(channel)

    def forward(self, x):
        # x = torch.cat([x_1, x_edge], dim=1)
        out = F.relu(self.bn_1(self.conv_1(x)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        out = F.relu(self.bn_3(self.conv_3(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Boundary Refinement Module
class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


class BASNet(nn.Module):
    def __init__(self, cfg):
        super(BASNet, self).__init__()
        self.cfg = cfg
        block = BasicBlock
        self.bkbone = ResNet(block, [2, 2, 2, 2])
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.path1_1 = nn.Sequential(
            conv1x1(512 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_2 = nn.Sequential(
            conv1x1(512 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_3 = nn.Sequential(
            conv1x1(256 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.path2 = nn.Sequential(
            conv3x3(128 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.path3 = nn.Sequential(
            conv1x1(64 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.fuse1_1 = CBM(64)
        self.fuse1_2 = CBM(64)
        self.fuse12 = CBM(64)
        self.fuse3 = CBM(64)
        self.fuse23 = BRM(64)

        self.head_1 = conv3x3(64, 1, bias=True)
        self.head_2 = conv3x3(64, 1, bias=True)
        self.head_3 = conv3x3(64, 1, bias=True)
        self.head_4 = conv3x3(64, 1, bias=True)
        self.head_5 = conv3x3(64, 1, bias=True)
        self.head_edge = conv3x3(64, 1, bias=True)

        self.initialize()

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        l1, l2, l3, l4, l5 = self.bkbone(x)

        path1_1 = F.avg_pool2d(l5, l5.size()[2:])
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=l5.size()[2:], mode='bilinear',
                                align_corners=True)  # 1/32  # 双线性插值上采样，使得与l5相同
        path1_2 = F.relu(self.path1_2(l5), inplace=True)  # 1/32
        path1_2 = RCF(path1_1, path1_2)
        path1_2 = self.fuse1_1(path1_2)  # 1/32
        path1_2 = F.interpolate(path1_2, size=l4.size()[2:], mode='bilinear', align_corners=True)  # 1/16

        path1_3 = F.relu(self.path1_3(l4), inplace=True)  # 1/16
        path1 = RCF(path1_2, path1_3)
        path1 = self.fuse1_2(path1)  # 1/16
        path1 = F.interpolate(path1, size=l3.size()[2:], mode='bilinear', align_corners=True)

        l3 = self.cbam2(l3)
        path2 = self.path2(l3)  # 1/8 #SAM
        path12 = RCF(path1, path2)
        path12 = self.fuse12(path12)  # 1/8
        path12 = F.interpolate(path12, size=l2.size()[2:], mode='bilinear', align_corners=True)  # 1/4

        l2 = self.cbam1(l2)
        path3_1 = l2  # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        path3 = RCF(path3_1, path3_2)
        path3 = self.fuse3(path3)  # 1/4

        path_out = self.fuse23(path12, path3)  # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)

        if self.cfg.mode == 'train':
            logits_2 = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)
            logits_3 = F.interpolate(self.head_3(path1), size=shape, mode='bilinear', align_corners=True)

            return logits_1, logits_2, logits_3
        else:
            return logits_1

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
