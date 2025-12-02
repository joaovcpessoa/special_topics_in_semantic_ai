import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

HPARAMS = {
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'BATCH_SIZE': 128,
    'Z_DIM': 100,
    'LEARNING_RATE_G': 1e-4,
    'LEARNING_RATE_D': 1e-4,
    'NUM_EPOCHS': 50,
    'CHANNELS_IMG': 1,
    'FEATURES_D': 64,
    'FEATURES_G': 64,
    'CRITIC_ITERATIONS': 5,
    'CLIP_VALUE': 0.01
}

# DATALOADER
def get_mnist_dataloader(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    fixed_noise = torch.randn(64, HPARAMS['Z_DIM'], 1, 1).to(HPARAMS['DEVICE'])
    return dataloader, fixed_noise

## GAN
class GeneratorGAN(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, img_dim),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class DiscriminatorGAN(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
    
## DCGAN / WGAN
class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g * 4, 4, 1, 0),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 3, 2, 1, 1),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorDCGAN(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 2, features_d * 4, 3, 2, 1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

class CriticWGAN(DiscriminatorDCGAN):
    def __init__(self, channels_img, features_d):
        super().__init__(channels_img, features_d)
        self.net[-1] = nn.Sequential(
            nn.Conv2d(features_d * 4, 1, 4, 1, 0),
        )
        
## PLOT
def plot_generated_images(generator, fixed_noise, z_dim, device, epoch, model_name, path='./results'):
    os.makedirs(path, exist_ok=True)
    
    if model_name == 'GAN':
        noise = fixed_noise.view(fixed_noise.size(0), -1)
    else:
        noise = fixed_noise

    generator.eval()
    with torch.no_grad():
        generated_images = generator(noise).detach().cpu()
        
    generator.train()
    
    generated_images = 0.5 * generated_images + 0.5 

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.suptitle(f"{model_name} - Época {epoch}", fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        img = generated_images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.savefig(f'{path}/{model_name}_epoch_{epoch:03d}.png')
    plt.close(fig)

def plot_losses(all_losses, model_name):
    epochs = range(1, len(all_losses['G']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_losses['G'], label=f'{model_name} Gerador Loss')
    d_c_key = 'C' if model_name == 'WGAN' else 'D'
    d_c_label = 'Crítico Loss (Dist. Wasserstein)' if model_name == 'WGAN' else 'Discriminador Loss'
    
    plt.plot(epochs, all_losses[d_c_key], label=f'{model_name} {d_c_label}')
    
    plt.title(f"Perdas de Treinamento - {model_name}")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/{model_name}_losses.png')
    plt.close()
    print(f"Perdas de {model_name} salvas em ./results/{model_name}_losses.png")
    
def train_gan_dcgan(G, D, dataloader, fixed_noise, model_name):
    optimizerG = optim.Adam(G.parameters(), lr=HPARAMS['LEARNING_RATE_G'], betas=(0.5, 0.999))
    optimizerD = optim.Adam(D.parameters(), lr=HPARAMS['LEARNING_RATE_D'], betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    all_losses = defaultdict(list)
    real_label, fake_label = 1.0, 0.0

    for epoch in range(1, HPARAMS['NUM_EPOCHS'] + 1):
        loss_G_epoch, loss_D_epoch = 0, 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(HPARAMS['DEVICE'])
            b_size = data.size(0)
            D.zero_grad()
            label_real = torch.full((b_size, 1), real_label, dtype=torch.float, device=HPARAMS['DEVICE'])
            output_real = D(data)
            errD_real = criterion(output_real, label_real)
            noise = torch.randn(b_size, HPARAMS['Z_DIM'], 1, 1).to(HPARAMS['DEVICE'])
            if model_name == 'GAN':
                noise = noise.view(b_size, HPARAMS['Z_DIM'])
            
            fake = G(noise).detach()
            label_fake = torch.full((b_size, 1), fake_label, dtype=torch.float, device=HPARAMS['DEVICE'])
            output_fake = D(fake)
            errD_fake = criterion(output_fake, label_fake)

            errD = (errD_real + errD_fake) / 2
            errD.backward()
            optimizerD.step()

            G.zero_grad()
            output_g = D(G(noise))
            errG = criterion(output_g, label_real)
            errG.backward()
            optimizerG.step()
            
            loss_D_epoch += errD.item()
            loss_G_epoch += errG.item()
            num_batches += 1

        avg_D_loss = loss_D_epoch / num_batches
        avg_G_loss = loss_G_epoch / num_batches
        all_losses['D'].append(avg_D_loss)
        all_losses['G'].append(avg_G_loss)
        
        print(f'Epoch [{epoch}/{HPARAMS["NUM_EPOCHS"]}] | D Loss: {avg_D_loss:.4f} | G Loss: {avg_G_loss:.4f}')
        
        if epoch % 5 == 0 or epoch == HPARAMS['NUM_EPOCHS']:
            plot_generated_images(G, fixed_noise, HPARAMS['Z_DIM'], HPARAMS['DEVICE'], epoch, model_name)
    
    plot_losses(all_losses, model_name)
    return G, D

def train_wgan(G, C, dataloader, fixed_noise):
    optimizerG = optim.RMSprop(G.parameters(), lr=HPARAMS['LEARNING_RATE_G'])
    optimizerC = optim.RMSprop(C.parameters(), lr=HPARAMS['LEARNING_RATE_D'])
    
    all_losses = defaultdict(list)
    
    for epoch in range(1, HPARAMS['NUM_EPOCHS'] + 1):
        loss_C_epoch, loss_G_epoch = 0, 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(HPARAMS['DEVICE'])
            b_size = data.size(0)
            
            for _ in range(HPARAMS['CRITIC_ITERATIONS']):
                C.zero_grad()
                
                C_real = C(data).view(-1)
                
                noise = torch.randn(b_size, HPARAMS['Z_DIM'], 1, 1).to(HPARAMS['DEVICE'])
                fake = G(noise).detach() 
                C_fake = C(fake).view(-1)
                
                errC = -(torch.mean(C_real) - torch.mean(C_fake))
                
                errC.backward()
                optimizerC.step()
                
                for p in C.parameters():
                    p.data.clamp_(-HPARAMS['CLIP_VALUE'], HPARAMS['CLIP_VALUE'])

            loss_C_epoch += errC.item()

            G.zero_grad()
            noise = torch.randn(b_size, HPARAMS['Z_DIM'], 1, 1).to(HPARAMS['DEVICE'])
            fake = G(noise) 
            C_fake = C(fake).view(-1)
            
            errG = -torch.mean(C_fake)
            errG.backward()
            optimizerG.step()
            
            loss_G_epoch += errG.item()
            num_batches += 1

        avg_C_loss = loss_C_epoch / (num_batches * HPARAMS['CRITIC_ITERATIONS'])
        avg_G_loss = loss_G_epoch / num_batches
        all_losses['C'].append(avg_C_loss)
        all_losses['G'].append(avg_G_loss)
        
        print(f'Epoch [{epoch}/{HPARAMS["NUM_EPOCHS"]}] | C Loss (Wasserstein): {avg_C_loss:.4f} | G Loss: {avg_G_loss:.4f}')
        
        if epoch % 5 == 0 or epoch == HPARAMS['NUM_EPOCHS']:
            plot_generated_images(G, fixed_noise, HPARAMS['Z_DIM'], HPARAMS['DEVICE'], epoch, "WGAN")

    plot_losses(all_losses, "WGAN")
    return G, C

## main
def main():
    os.makedirs('./results', exist_ok=True)
    dataloader, fixed_noise = get_mnist_dataloader(HPARAMS['BATCH_SIZE'])

    IMG_DIM_FLAT = 28 * 28
    G_gan = GeneratorGAN(HPARAMS['Z_DIM'], IMG_DIM_FLAT).to(HPARAMS['DEVICE'])
    D_gan = DiscriminatorGAN(IMG_DIM_FLAT).to(HPARAMS['DEVICE'])
    train_gan_dcgan(G_gan, D_gan, dataloader, fixed_noise, "GAN")

    G_dcgan = GeneratorDCGAN(HPARAMS['Z_DIM'], HPARAMS['CHANNELS_IMG'], HPARAMS['FEATURES_G']).to(HPARAMS['DEVICE'])
    D_dcgan = DiscriminatorDCGAN(HPARAMS['CHANNELS_IMG'], HPARAMS['FEATURES_D']).to(HPARAMS['DEVICE'])
    train_gan_dcgan(G_dcgan, D_dcgan, dataloader, fixed_noise, "DCGAN")

    G_wgan = GeneratorDCGAN(HPARAMS['Z_DIM'], HPARAMS['CHANNELS_IMG'], HPARAMS['FEATURES_G']).to(HPARAMS['DEVICE'])
    C_wgan = CriticWGAN(HPARAMS['CHANNELS_IMG'], HPARAMS['FEATURES_D']).to(HPARAMS['DEVICE'])
    train_wgan(G_wgan, C_wgan, dataloader, fixed_noise)
    print("Fim")

if __name__ == '__main__':
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)       
    main()