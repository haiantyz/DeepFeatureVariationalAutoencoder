import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

from modules import my_vgg, ae_module, loss_module
import os

###

learning_rate = 0.001
batch_size = 64

epochs = 5

content_layers = ['r11', 'r21', 'r31']

cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print_every = 500

###

data_dir = '/media/peter/HDD 1/datasets_peter/CelebA/Img'

prep_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=prep_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

###

auto_encoder = ae_module.Auto_Encoder(input_channels=3, bn_momentum=0.9)
vgg = my_vgg.VGG()
content_loss = loss_module.Content_Loss()

for param in vgg.parameters():
    param.requires_grad = False

if cuda:
    auto_encoder.cuda()
    vgg.cuda()
    content_loss.cuda()

###

adam = torch.optim.Adam(auto_encoder.parameters(), lr=learning_rate)

###

for e in range(epochs):
    print('\n\nEpoch {} of {}'.format(e, epochs))

    auto_encoder.train()
    loss_counter = 0.
    for i, (images, _) in enumerate(train_loader):
        if i % print_every == 0: print('Batch {} of {}'.format(i, len(train_loader)))
        images = Variable(images.type(dtype))
        adam.zero_grad()
        target = vgg(images, out_keys=content_layers)
        reconstruced, mean, logvar = auto_encoder(images, out_keys=['reconstructed', 'mean', 'logvar'])
        output = vgg(reconstruced, out_keys=content_layers)
        loss = content_loss(output, target, mean, logvar)
        loss_counter += loss.data
        loss.backward()
        adam.step()
    print('Average loss over epoch = {}'.format(loss_counter / (i + 1)))
    os.makedirs('./models', exist_ok=True)
    torch.save(auto_encoder.state_dict(), './ae_params_epoch{}.pt'.format(e))
