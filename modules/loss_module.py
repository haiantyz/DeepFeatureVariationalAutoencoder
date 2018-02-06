import torch.nn as nn
import torch


class Content_Loss(nn.Module):

    def __init__(self, alpha=1, beta=0.5):
        super(Content_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = nn.MSELoss()

    def forward(self, output, target, mean, logvar):
        kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())  # or should we use torch.sum() ?
        loss_list = [self.criterion(output[layer], target[layer]) for layer in range(len(output))]
        content = sum(loss_list)
        return self.alpha * kld + self.beta * content


###

class Plain_Loss(nn.Module):

    def __init__(self, kld_weight=0.01):
        super(Plain_Loss, self).__init__()
        self.criterion = nn.MSELoss()
        self.kld_weight = kld_weight

    def forward(self, output, target, mean, logvar):
        kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp()) # or should we use torch.sum() ?
        content = self.criterion(output, target)
        return self.kld_weight * kld + content
