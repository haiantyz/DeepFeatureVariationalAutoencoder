import torch.nn as nn
import torch


# def sum_list(x):
#     for i in range(len(x)):
#         if i == 0:
#             out = x[i]
#         else:
#             out += x[i]
#     return out


###


class Content_Loss(nn.Module):

    def __init__(self, alpha=1, beta=0.5):
        super(Content_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = nn.MSELoss()

    def forward(self, output, target, mean, logvar):
        # KLD mean
        KLD = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # # KLD sum
        # KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        # KLD = torch.mean(KLD, dim=0)

        loss_list = [self.criterion(output[layer], target[layer]) for layer in range(len(output))]
        content = sum(loss_list)

        return self.alpha * KLD + self.beta * content
