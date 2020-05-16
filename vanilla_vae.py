import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaVae(nn.Module):

    def __init__(self):
        '''
        hidden layers number: 1
        neurons number of the hidden layer: 256
        neurons number of the latent layer: 100
        '''

        super(VanillaVae, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, 100)
        self.var = nn.Linear(256, 100)

        self.decoder = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def encode(self, X):
        h = self.encoder(X)
        z_mu = self.mu(h)
        z_logvar = self.var(h)
        return z_mu, z_logvar

    def decode(self, z):
        X = self.decoder(z)
        return X

    def sample(self, mu, logvar):
        '''
        reparameterization trick
        the sampling process is outside of the network
        so the gradient won’t flow through it
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = mu + std * eps
        return z

    def forward(self, X):
        z_mu, z_logvar = self.encode(X)
        z_sample = self.sample(z_mu, z_logvar)
        X_sample = self.decode(z_sample)
        return X_sample, z_sample, z_mu, z_logvar
    
    def loss(self, X, X_sample, Z_sample, z_mu, z_logvar):
        '''
        average loss = reconstruction loss + KL loss for each data in minibatch
        '''
        recons_loss = F.binary_cross_entropy(X, X_sample)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_logvar) + z_mu ** 2 - 1 - z_logvar, 1))
        return recons_loss + kl_loss

    def generate(self, X):
        '''
        generate X_sample
        '''
        return self.forward(X)[0]


def train(net, loader):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(10000):
        for index, data in enumerate(loader):
            X, _ = data     # TODD: view
            optimizer.zero_grad()
            X_sample, z_sample, z_mu, z_logvar = net(X)
            loss = net.loss(X, X_sample, z_sample, z_mu, z_logvar)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('Epoch: %3d\tBatch: %3d\t| average loss: %.4f' % (epoch + 1, index + 1, loss))

