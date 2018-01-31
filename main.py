import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import ae_module
import my_vgg

###

learning_rate = 0.001
batch_size = 64

content_layers = ['r11', 'r21', 'r31']

cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

###

data_dir = '/media/peter/HDD 1/datasets_peter/CelebA/Img'

prep_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=prep_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

###

auto_encoder = ae_module.auto_encoder()
vgg = my_vgg.vgg()
if cuda:
    auto_encoder.cuda()
    vgg.cuda()

for i, (x, _) in enumerate(train_loader):
    x = Variable(x.type(dtype))
    y = auto_encoder(x)
    target = vgg(x, content_layers)
    output = vgg(x, content_layers)

    c = 2
