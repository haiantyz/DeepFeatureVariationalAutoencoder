import torch
import torch.nn as nn
from torch.autograd import Variable


def conv4x4_bn_relu(channels_in, channels_out, bn_momentum):
    return nn.Sequential(
        nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(num_features=channels_out, momentum=bn_momentum),
        nn.LeakyReLU(negative_slope=0.01)
    )


def up_conv3x3_bn_relu(channels_in, channels_out, bn_momentum):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=channels_out, momentum=bn_momentum),
        nn.LeakyReLU(negative_slope=0.01)
    )


###

class Encoder(nn.Module):

    def __init__(self, input_channels=3, bn_momentum=0.9):
        super(Encoder, self).__init__()
        self.bn0 = nn.BatchNorm2d(num_features=input_channels, momentum=bn_momentum)
        self.encode_layer1 = conv4x4_bn_relu(input_channels, 32, bn_momentum)
        self.encode_layer2 = conv4x4_bn_relu(32, 64, bn_momentum)
        self.encode_layer3 = conv4x4_bn_relu(64, 128, bn_momentum)
        self.encode_layer4 = conv4x4_bn_relu(128, 256, bn_momentum)
        self.mean_layer = nn.Linear(in_features=256 * 4 * 4, out_features=100)
        self.logvar_layer = nn.Linear(in_features=256 * 4 * 4, out_features=100)

    def forward(self, x):
        x = self.bn0(x)
        x = self.encode_layer1(x)
        x = self.encode_layer2(x)
        x = self.encode_layer3(x)
        x = self.encode_layer4(x)
        x = x.view(x.size(0), -1)
        return self.mean_layer(x), self.logvar_layer(x)


class Decoder(nn.Module):

    def __init__(self, bn_momentum=0.9):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=100, out_features=256 * 4 * 4),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.decode_layer1 = up_conv3x3_bn_relu(256, 128, bn_momentum)
        self.decode_layer2 = up_conv3x3_bn_relu(128, 64, bn_momentum)
        self.decode_layer3 = up_conv3x3_bn_relu(64, 32, bn_momentum)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 256, 4, 4)
        z = self.decode_layer1(z)
        z = self.decode_layer2(z)
        z = self.decode_layer3(z)
        return self.final(z)


###


class Auto_Encoder(nn.Module):

    def __init__(self, input_channels=3, bn_momentum=0.9):
        super(Auto_Encoder, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, bn_momentum=bn_momentum)
        self.decoder = Decoder(bn_momentum=bn_momentum)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        std = 0.5 * torch.exp(logvar)
        eps = Variable(torch.normal(torch.zeros_like(std.data), torch.ones_like(std.data)))
        z = mean + eps * std
        return self.decoder(z), mean, logvar
