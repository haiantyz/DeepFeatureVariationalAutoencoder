# Deep Feature Consistent Variational Autoencoder

Reproduction of the results of [this paper](http://ieeexplore.ieee.org/document/7926714/) on 'Deep Feature Consistent Variational Autoencoders' in PyTorch.

*Input images*

<img src='results/input.png' width='100%'/>

*Reconstruced images using content loss from VGG layers relu1_1, relu2_1, relu3_1.*

<img src='results/reconstructed.png' width='100%'/>

*And using a plain VAE (PVAE).* Note that the ratio of KLD loss to reconstruction loss is important!

<img src='results/reconstructed_plain.png' width='100%'/>

# Links

[arxiv](https://arxiv.org/abs/1610.00291)