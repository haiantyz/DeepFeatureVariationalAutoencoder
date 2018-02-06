import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from modules import ae_module
import os

###

batch_size = 16

cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

###

data_dir = '/media/peter/HDD 1/datasets_peter/CelebA/Img'

prep_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=prep_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

###

auto_encoder = ae_module.Auto_Encoder(input_channels=3, bn_momentum=0.9)
auto_encoder.load_state_dict(torch.load('./models/ae_params_epoch4_plain.pt'))
if cuda:
    auto_encoder.cuda()

###

auto_encoder.eval()
for i, (images, _) in enumerate(train_loader):
    images = Variable(images.type(dtype))
    reconstruced, _, _ = auto_encoder(images)
    break

###

os.makedirs('./results', exist_ok=1)
save_image(images.data, './results/input.png')
save_image(reconstruced.data, './results/reconstructed_plain.png')
