##########################
### MODEL
##########################
import torch.nn as nn
import torch.nn.functional as F
import torch
from util import hierarchy_dict, get_index, get_index_for_model_2





def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, NUM_level_1, NUM_level_2, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc_6 = nn.Linear(2048 * block.expansion, NUM_level_2)  #2048*4 = 8192
        # self.fc_61 =nn.Linear(64, num_classes-6)  #2048*4 = 8192

        self.avgpool0 = nn.AvgPool2d(6, stride=1, padding=0)
        self.fc0 = nn.Linear(9216, 32)
        self.fc01 = nn.Linear(32, NUM_level_1)


        # self.fc1 = nn.Linear(2048 * block.expansion, num_classes[1])
        # self.fc2 = nn.Linear(2048 * block.expansion, num_classes[2])
        # self.fc3 = nn.Linear(2048 * block.expansion, num_classes[3])
        # self.fc4 = nn.Linear(2048 * block.expansion, num_classes[4])
        # self.fc5 = nn.Linear(2048 * block.expansion, num_classes[5])
        # self.fc6 = nn.Linear(2048 * block.expansion, num_classes[6])

        # 最后的类别的Index
        self.idx_list = get_index_for_model_2(hierarchy_dict)  #[2, 7, 13, 17, 25, 34, 34, 34, 34, 34]






        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
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
        x3 = self.layer3(x)
        x4 = self.layer4(x3)

        #给6个head的 feature
        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)

        #给第一个Head的feature
        x0 = self.avgpool0(x3)
        x0 = x0.view(x0.size(0), -1)


        #2层fc
        logits = self.fc_6(x)
        # logits = self.relu(logits)
        # logits = self.fc_61(logits)


        logits_0 = self.fc0(x0)
        logits_0 = self.relu(logits_0)
        logits_0 = self.fc01(logits_0)



        logits_1 = logits[:, 0: self.idx_list[0]]
        logits_2 = logits[:, self.idx_list[0]:self.idx_list[1]]
        logits_3 = logits[:, self.idx_list[1]:self.idx_list[2]]
        logits_4 = logits[:, self.idx_list[2]:self.idx_list[3]]
        logits_5 = logits[:, self.idx_list[3]:self.idx_list[4]]
        logits_6 = logits[:, self.idx_list[4]:]


        # logits_1 = self.fc1(x)
        # logits_2 = self.fc2(x)
        # logits_3 = self.fc3(x)
        # logits_4 = self.fc4(x)
        # logits_5 = self.fc5(x)
        # logits_6 = self.fc6(x)



        probas_0 = F.softmax(logits_0, dim=1) #第1个Head是 Level-1，6个group
        probas_1 = F.softmax(logits_1, dim=1) * probas_0[:,0:1] #第2Head是'Skates'的 2 类
        probas_2 = F.softmax(logits_2, dim=1) * probas_0[:,1:2] #第3Head是'Sharks'的 4 类
        probas_3 = F.softmax(logits_3, dim=1) * probas_0[:,2:3]
        probas_4 = F.softmax(logits_4, dim=1) * probas_0[:,3:4]
        probas_5 = F.softmax(logits_5, dim=1) * probas_0[:,4:5]
        probas_6 = F.softmax(logits_6, dim=1) * probas_0[:,5:6]


        probas_level2 = torch.cat((probas_1,probas_2,probas_3,probas_4,probas_5,probas_6), dim=1)



        return (logits_0, logits_1,logits_2,logits_3,logits_4,logits_5,logits_6),\
               (probas_0,probas_1,probas_2,probas_3,probas_4,probas_5,probas_6), \
               probas_level2



def resnet101(NUM_level_1_CLASSES,  NUM_level_2_CLASSES, grayscale): # NUM_CLASSES is a List [6, 2,3,6,2,8,9]
    """Constructs a ResNet-50 model."""
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   NUM_level_1=NUM_level_1_CLASSES,
                   NUM_level_2=NUM_level_2_CLASSES,
                   grayscale=grayscale)
    return model