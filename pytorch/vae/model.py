import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VaeNet(nn.Module):
    """
    This class will have the encoder and the decoder networks
    of the variational autoencoder. The encoder will take the 
    input image and and transform the image to it's latent space.
    There will also be a separate method related to the ELBO loss
    which will be calculated as follows:

    Elbo(x) = Marginal_likelihood(x) - KL_divergence(posterior || true_prior)

    Applying the same model to both the MNIST dataset and the
    CIFAR10 dataset. So the things should be agnostic with each 
    others those that deal with the dimensionality of things.

    Since we are using the same model for both the datasets, we
    should keep the capasity as high as possible so that it is
    capable enough to deal with the complexity of the tougher
    dataset i.e. CIFAR10.
    """

    def __init__(self, latent_dim, batch_size):
        super(VaeNet, self).__init__() 
        
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.encoder_network()
        self.decoder_network()
        

    def encoder_network(self):
        # The definition of the convolutional layer takes in the
        # number of channels of the input matrix, the number of 
        # channels for the output matrix, the size of the 
        # convolutional kernel.
        self.en_conv1 = nn.Conv2d(3, 8, 3, padding=1) 
        self.en_conv2 = nn.Conv2d(8, 16, 3)
        self.en_bn2 = nn.BatchNorm2d(16)
        self.en_conv3 = nn.Conv2d(16, 32, 2, stride=2) # [BSx32x15x15]
        self.en_bn3 = nn.BatchNorm2d(32)
        self.en_conv4 = nn.Conv2d(32, 64, 3, stride=2) 
        self.en_bn4 = nn.BatchNorm2d(64)
        self.en_conv5 = nn.Conv2d(64, 128, 3, stride=2) # [BSx128x3x3]
        self.en_bn5 = nn.BatchNorm2d(128)
        self.en_fc1 = nn.Linear(128 * 3 * 3, 256)
        self.en_mu = nn.Linear(256, self.latent_dim)
        self.en_sigma = nn.Linear(256, self.latent_dim)

    def decoder_network(self):
        self.de_deconv1 = nn.ConvTranspose2d(self.latent_dim, self.batch_size * 4, 4, 1, 0)
        self.de_bn1 = nn.BatchNorm2d(self.batch_size * 4)
        self.de_deconv2 = nn.ConvTranspose2d(self.batch_size * 4, self.batch_size * 2, 4, 2, 1)
        self.de_bn2 = nn.BatchNorm2d(self.batch_size * 2)
        self.de_deconv3 = nn.ConvTranspose2d(self.batch_size * 2, self.batch_size, 4, 2, 1)
        self.de_bn3 = nn.BatchNorm2d(self.batch_size)
        self.de_deconv4 = nn.ConvTranspose2d(self.batch_size, 3, 4, 2, 1)

    def encoder(self, x):
        x = F.elu(self.en_conv1(x))
        x = F.elu(self.en_bn2(self.en_conv2(x)))
        x = F.elu(self.en_bn3(self.en_conv3(x)))
        x = F.elu(self.en_bn4(self.en_conv4(x)))
        x = F.elu(self.en_bn5(self.en_conv5(x)))
        x = x.view(-1, 128 * 3 *3) # flatten
        x = F.elu(self.en_fc1(x))
        x_mu = self.en_mu(x)
        x_log_sigma_sq = self.en_sigma(x)
        return x_mu, x_log_sigma_sq

    def reparameterize(self, mu, log_sigma_sq):
        if self.training:
            std = log_sigma_sq.mul(0.5).exp_() # Doing things in place
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu) # multiply the std with epsilon and add it to the mean
        else:
            return mu

    def decoder(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        x = F.elu(self.de_bn1(self.de_deconv1(x)))
        x = F.elu(self.de_bn2(self.de_deconv2(x)))
        x = F.elu(self.de_bn3(self.de_deconv3(x)))
        x = F.tanh((self.de_deconv4(x)))
        return x
    
    def forward(self,x):
       mu, log_sigma_sq = self.encoder(x)
       z = self.reparameterize(mu, log_sigma_sq)
       reconstructed_img = self.decoder(z)
       return reconstructed_img, mu, log_sigma_sq


