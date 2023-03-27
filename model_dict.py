import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def get_block_repetitions_for_original_resnet( depth, block_name ):
    if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
    elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
    else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
    return n, block

class OriginalResNet(nn.Module):
    def __init__(self, depth, block_name, num_classes=100, all_planes=[16, 16, 32, 64], ):
        super(OriginalResNet, self).__init__()

        n, block = get_block_repetitions_for_original_resnet( depth, block_name )

        self.in_planes = all_planes[0] #64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, all_planes[1], n, stride=1)
        self.layer2 = self._make_layer(block, all_planes[2], n, stride=2)
        self.layer3 = self._make_layer(block, all_planes[3], n, stride=2)
        self.linear = nn.Linear( all_planes[3] *block.expansion, num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.conv_channels = [ all_planes[2]  ]
        self.xchannels   = [ all_planes[3] *block.expansion ] 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return self.xchannels[-1]

    def get_message(self):
        return 'Original ResNet (CIFAR)'#self.message

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        ft = [ out ]

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        features  = out 

        out = self.linear(out)
        logits = out
        return features, logits, ft

class NewResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, all_planes=[64, 128, 256, 512]):
        super(NewResNet, self).__init__()
        self.in_planes = all_planes[0] #64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, all_planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, all_planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, all_planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear( all_planes[3] *block.expansion, num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.xchannels   = [ all_planes[3] *block.expansion ] 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return self.xchannels[-1]

    def get_message(self):
        return 'ResNet (CIFAR)'#self.message

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        ft = [ out ]

        out = self.layer4(out)
        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)
        features  = out 

        out = self.linear(out)
        logits = out
        return features, logits, ft

def ResNet18(num_classes=10):
    return NewResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes=10, adaptive_pool=False):
    return NewResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def ResNet50(num_classes=10, adaptive_pool=False):
    return NewResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def ResNet32(num_classes=10):
    return OriginalResNet(32, 'basicblock', num_classes=num_classes, all_planes=[16, 16, 32, 64], )

def get_model_from_name( num_classes, name ):
    model_dict = {}
    model_dict['ResNet18'] = ResNet18
    model_dict['ResNet34'] = ResNet34
    model_dict['ResNet50'] = ResNet50
    model_dict['ResNet32'] = ResNet32

    if name in model_dict:
        model = model_dict[ name ]( num_classes )
    else:
        assert('Model not defined.')
    return model

def get_model_infos( model, shape, cuda=True ): 
    inputs = torch.randn( *shape )
    if cuda: inputs = inputs.cuda()

    macs, params = profile( model, inputs=(inputs,), verbose=False )
    macs = macs / 1e6
    params = params / 2 ** 20
    return macs, params

def test( name ):
    import numpy as np
    dataset = 'CIFAR-100'
    #dataset = 'aug-tiny-imagenet-200'
    if dataset == 'CIFAR-100':
        num_classes=100
        xshape = (1,3,32, 32)
    elif dataset == 'aug-tiny-imagenet-200':
        num_classes=200
        xshape = (1,3,64, 64)
    #net = ResNet50(num_classes)
    #net = ResNet34(num_classes)
    #net = ResNet18(num_classes)
    #net = get_model_from_name( num_classes, name, 'CIFAR-100' )
    net = get_model_from_name( num_classes, name) #, dataset )

    #from utils import get_model_infos

    flop, param = get_model_infos(net, xshape, cuda=False)

    print(' Model -- ' + name )
    print(
        "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )
    print()

    counts = sum(np.prod(v.size()) for v in net.parameters())
    print(counts)


'''
test( 'ResNet18' )
test( 'ResNet50' )
test( 'ResNet32' )
#'''
