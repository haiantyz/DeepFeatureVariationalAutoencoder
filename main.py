import torch
from torchvision import datasets, transforms

data_dir = '/media/peter/HDD 1/datasets_peter/CelebA/Img/img_align_celeba'

prep_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=prep_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for i, (x, _) in enumerate(train_loader):
    print(i)
