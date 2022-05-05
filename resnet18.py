import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}
import train


class MyConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride=2):
        super(MyConv, self).__init__()
        self.outchannels = outchannels
        self.inchannels = inchannels
        self.stride = stride
        self.weight = torch.Tensor(outchannels, inchannels, 3, 5)
        self.weight = nn.Parameter(data=self.weight, requires_grad=True)
        kernel = train.get_kernel_shape()
        if kernel == "only_ellipse":
            kernel_weight = [[0.34, 0.86, 0.99, 0.86, 0.34],
                             [0.95, 1.0, 1.0, 1.0, 0.95],
                             [0.34, 0.86, 0.99, 0.86, 0.34]
                             ]
        if kernel == "square_ellipse":
            kernel_weight = [[0.67, 0.93, 0.995, 0.93, 0.67],
                             [0.975, 1.0, 1.0, 1.0, 0.975],
                             [0.67, 0.93, 0.995, 0.93, 0.67]]
        if kernel == "both":
            kernel_weight = [[1.34, 1.86, 1.99, 1.86, 1.34],
                             [1.95, 2.0, 2.0, 2.0, 1.95],
                             [1.34, 1.86, 1.99, 1.86, 1.34]
                             ]
        self.w1 = torch.Tensor([[
            kernel_weight
        ]])
        # self.reset_parameters()

    def forward(self, x):
        self.w1 = self.w1.cuda(x.device)
        x = F.conv2d(x, self.weight * self.w1, stride=self.stride, padding=(1, 2), groups=1)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = MyConv(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MyConv(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, MyConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MyConv(self.inplanes, planes * block.expansion,
                       stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
