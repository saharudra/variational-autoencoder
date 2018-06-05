from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VaeNet

# Define arguments required for training using parser.
parser = argparse.ArgumentParser(description='VAE CIFAR example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--latent_dim', type=int, default=100, metavar='L',
                    help='size of the latent dimension (default: 100)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait for before logging training status')

# Parse the arguments and see if cuda is available
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Use the defined seed to initialize state
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Define the transformation process of the data.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data/', train=True, download=True,
        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=3)
testset = datasets.CIFAR10(root='./data/', train=False, download=True, 
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=3)
# Define the model and port it to the gpu
model = VaeNet(batch_size=args.batch_size, latent_dim=args.latent_dim)
if args.cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the loss function.
def vae_loss(x_recons, x_original, mu, log_sigma_sq):
    reconstruct_loss = F.mse_loss(x_recons, x_original)
    # KL divergence loss can be defined as follows
    # 0.5 * sum(1 + log(sigma^2) - mu^2 -sigma^2)
    kl_div = -0.5 * torch.sum(1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())
    kl_div /= args.batch_size * 32 * 32 * 3
    return kl_div, reconstruct_loss

# Define the train step
def train(epoch):
    model.train()
    train_loss = 0
    likelihood = 0
    divergence = 0
    for batch_idx, data in enumerate(trainloader):
        images, labels = data
        if args.cuda:
            images = Variable(images.cuda())
        optimizer.zero_grad()
        reconstructed_img, mu, log_sigma_sq = model(images)
        kl_div, recon_loss = vae_loss(reconstructed_img, images, mu, log_sigma_sq)
        loss = kl_div + recon_loss
        loss.backward()
        train_loss += loss.data[0]
        likelihood += recon_loss
        divergence += kl_div
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.data[0] / len(data)))
    print('Epoch: %f, Average loss: %f, Average reconstruction loss: %f, Average kl divergence loss: %f' \
           % (epoch, train_loss / len(trainloader.dataset), \
           likelihood / len(trainloader.dataset), \
           divergence / len(trainloader.dataset)))

# Define the test step
def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(testloader):
        images, labels = data
        if args.cuda:
            images = Variable(images.cuda())
        reconstructed_img, mu, log_sigma_sq = model(images)
        kl_div, recon_loss = vae_loss(reconstructed_img, images, mu, log_sigma_sq)
        test_loss += (kl_div + recon_loss).data[0]
        if batch_idx == 0:
            n = min(images.size(0), 8)
            comparison = torch.cat([images[:n], 
                                   reconstructed_img.view(args.batch_size, 3, 32, 32)[:n]])
            save_image(comparison.data, 
                        'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(testloader.dataset)
    print('Test set loss: %f' % (test_loss))

# Set up the training loop
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    # Sample a random value from the gaussian distribution
    sample = Variable(torch.randn(args.batch_size, 100))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decoder(sample)
    save_image(sample.data.view(args.batch_size, 3, 32, 32), 
            'results/sample_' + str(epoch) + '.png')

