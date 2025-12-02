import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

data_dim = 2
noise_dim = 1
hidden_dim = 128
batch_size = 64
num_epochs = 5000
learning_rate = 0.0001

# Target distribution
def real_data_sampler(num_samples):
    x = np.random.uniform(-3, 3, size=(num_samples, 1))
    noise = np.random.normal(0, 0.1, size=(num_samples, 1))
    y = np.sin(x) + noise
    
    data = np.hstack((x, y))
    return torch.tensor(data, dtype=torch.float32)

# G
class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, z):
        return self.net(z)

# D
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() # (0,1)
        )

    def forward(self, x):
        return self.net(x)

# Noise
def noise_sampler(num_samples):
    return torch.randn(num_samples, noise_dim)

G = Generator(noise_dim, data_dim, hidden_dim)
D = Discriminator(data_dim, hidden_dim)

criterion = nn.BCELoss()
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate / 2) 
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate)

# Plot
def plot_evolution(checkpoints):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    num_samples = 1000
    
    real_data = real_data_sampler(num_samples).numpy()
    
    for i, epoch in enumerate(checkpoints):
        G_checkpoint = Generator(noise_dim, data_dim, hidden_dim)
        checkpoint_path = os.path.join('gan_checkpoints', f'generator_epoch_{epoch}.pth')
        
        try:
            G_checkpoint.load_state_dict(torch.load(checkpoint_path))
            G_checkpoint.eval()
        except FileNotFoundError:
            continue

        z = noise_sampler(num_samples)
        fake_data = G_checkpoint(z).detach().numpy()
        
        ax = axes[i]
        ax.scatter(real_data[:, 0], real_data[:, 1], s=5, alpha=0.6, label='Real data')
        ax.scatter(fake_data[:, 0], fake_data[:, 1], s=5, alpha=0.6, color='red', label='Fake data')
        
        percent = int(epoch / num_epochs * 100)
        ax.set_title(f'Evolution of GAN - {percent}% ({epoch} Epochs)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    filename = 'gan_evolution_4_subplots.png'
    plt.savefig(filename)
    plt.close()
    
    return filename

# Train
def train_gan():
    D_losses = []
    G_losses = []
    
    checkpoints = [
        int(num_epochs * 0.25),
        int(num_epochs * 0.50),
        int(num_epochs * 0.75),
        num_epochs
    ]
    
    checkpoint_dir = 'gan_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        real_data = real_data_sampler(batch_size)
        real_labels = torch.ones(batch_size, 1)
        
        z = noise_sampler(batch_size)
        fake_data = G(z).detach()
        fake_labels = torch.zeros(batch_size, 1)
        
        all_data = torch.cat((real_data, fake_data))
        all_labels = torch.cat((real_labels, fake_labels))
        
        D.zero_grad()
        output = D(all_data)
        loss_D = criterion(output, all_labels)
        loss_D.backward()
        optimizer_D.step()
        
        D_losses.append(loss_D.item())

        z = noise_sampler(batch_size)
        fake_data = G(z)
        
        target_labels = torch.ones(batch_size, 1)
        
        G.zero_grad()
        output = D(fake_data)
        loss_G = criterion(output, target_labels)
        loss_G.backward()
        optimizer_G.step()
        
        G_losses.append(loss_G.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")
        
        if epoch in checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth')
            torch.save(G.state_dict(), checkpoint_path)
            print(f"Generator checkpoint")

    plt.figure(figsize=(10, 5)) 
    plt.plot(D_losses, label='Discriminator Loss') 
    plt.plot(G_losses, label='Generator Loss') 
    plt.title('GAN Training Losses') 
    plt.xlabel('Iteration (Batch)') 
    plt.ylabel('Loss') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig('gan_losses_evolution.png') 
    plt.close()

    return checkpoints

# Main
if __name__ == '__main__':
    checkpoints = train_gan()
    plot_evolution(checkpoints)