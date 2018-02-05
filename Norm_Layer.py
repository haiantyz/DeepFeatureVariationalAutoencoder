import torch
from torch.autograd import Variable


class ImageNet_Norm_Layer(torch.nn.Module):

    def __init__(self, image_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ImageNet_Norm_Layer, self).__init__()
        self.cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.mean = torch.zeros((3, image_size, image_size))
        self.std = torch.zeros((3, image_size, image_size))
        for i in range(3):
            self.mean[i, :, :] = mean[i]
            self.std[i, :, :] = std[i]
        self.mean = Variable(self.mean.type(dtype), requires_grad=0)
        self.std = Variable(self.std.type(dtype), requires_grad=0)

    def forward(self, input):
        return (input - self.mean) / self.std


###

class ImageNet_Norm_Layer_2(torch.nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ImageNet_Norm_Layer_2, self).__init__()
        self.cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.mean = Variable(torch.FloatTensor(mean).type(dtype), requires_grad=0)
        self.std = Variable(torch.FloatTensor(std).type(dtype), requires_grad=0)

    def forward(self, input):
        return ((input.permute(0, 2, 3, 1) - self.mean) / self.std).permute(0, 3, 1, 2)


###

# net = ImageNet_Norm_Layer_2()
# dummy = Variable(torch.rand((6, 3, 256, 256)).type(torch.cuda.FloatTensor))
# y = net(dummy)
#
# c = 2
