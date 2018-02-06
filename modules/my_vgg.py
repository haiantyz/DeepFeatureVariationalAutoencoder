from torchvision import models
import torch.nn as nn
import torch.nn.functional as functional
import torch
from torch.autograd import Variable

###

class ImageNet_Norm_Layer_2(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ImageNet_Norm_Layer_2, self).__init__()
        self.cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.mean = Variable(torch.FloatTensor(mean).type(dtype), requires_grad=0)
        self.std = Variable(torch.FloatTensor(std).type(dtype), requires_grad=0)

    def forward(self, input):
        return ((input.permute(0, 2, 3, 1) - self.mean) / self.std).permute(0, 3, 1, 2)


###

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.norm_layer = ImageNet_Norm_Layer_2()

        vgg = models.vgg19(pretrained=True)
        vgg_feats = vgg.features
        layers = list(vgg_feats.children())

        self.conv1_1 = layers.pop(0)
        layers.pop(0)
        self.conv1_2 = layers.pop(0)
        layers.pop(0)
        self.pool1 = layers.pop(0)

        self.conv2_1 = layers.pop(0)
        layers.pop(0)
        self.conv2_2 = layers.pop(0)
        layers.pop(0)
        self.pool2 = layers.pop(0)

        self.conv3_1 = layers.pop(0)
        layers.pop(0)
        self.conv3_2 = layers.pop(0)
        layers.pop(0)
        self.conv3_3 = layers.pop(0)
        layers.pop(0)
        self.conv3_4 = layers.pop(0)
        layers.pop(0)
        self.pool3 = layers.pop(0)

        self.conv4_1 = layers.pop(0)
        layers.pop(0)
        self.conv4_2 = layers.pop(0)
        layers.pop(0)
        self.conv4_3 = layers.pop(0)
        layers.pop(0)
        self.conv4_4 = layers.pop(0)
        layers.pop(0)
        self.pool4 = layers.pop(0)

        self.conv5_1 = layers.pop(0)
        layers.pop(0)
        self.conv5_2 = layers.pop(0)
        layers.pop(0)
        self.conv5_3 = layers.pop(0)
        layers.pop(0)
        self.conv5_4 = layers.pop(0)
        layers.pop(0)
        self.pool5 = layers.pop(0)

    def forward(self, x, out_keys):
        x = self.norm_layer(x)
        out = {}
        out['r11'] = functional.relu(self.conv1_1(x))
        out['r12'] = functional.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = functional.relu(self.conv2_1(out['p1']))
        out['r22'] = functional.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = functional.relu(self.conv3_1(out['p2']))
        out['r32'] = functional.relu(self.conv3_2(out['r31']))
        out['r33'] = functional.relu(self.conv3_3(out['r32']))
        out['r34'] = functional.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = functional.relu(self.conv4_1(out['p3']))
        out['r42'] = functional.relu(self.conv4_2(out['r41']))
        out['r43'] = functional.relu(self.conv4_3(out['r42']))
        out['r44'] = functional.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = functional.relu(self.conv5_1(out['p4']))
        out['r52'] = functional.relu(self.conv5_2(out['r51']))
        out['r53'] = functional.relu(self.conv5_3(out['r52']))
        out['r54'] = functional.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
