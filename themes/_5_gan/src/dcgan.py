import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configurações e Hiperparâmetros ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Hiperparâmetros
latent_dim = 100        # Dimensão do vetor de ruído de entrada (Z)
image_size = 28         # Imagens MNIST são 28x28
num_channels = 1        # Imagens MNIST são em escala de cinza (1 canal)
batch_size = 128
num_epochs = 50         # Menos épocas do que o VAE, mas você pode aumentar
lr = 0.0002             # Taxa de aprendizado
beta1 = 0.5             # Parâmetro para o otimizador Adam

# --- 2. Preparação dos Dados ---
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalização de -1 a 1 para o Discriminador (DCGAN original usa tanh na saída)
    transforms.Normalize((0.5,), (0.5,)),
])

# Download e carregamento do dataset MNIST
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. Definição do Modelo: Inicialização de Pesos ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- 4. Definição do Modelo: Gerador (Generator) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Entrada: Ruído Z de latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Tamanho do estado: 256 x 4 x 4

            nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Tamanho do estado: 128 x 7 x 7

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Tamanho do estado: 64 x 14 x 14

            nn.ConvTranspose2d(64, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Saída: num_channels x 28 x 28
        )

    def forward(self, input):
        return self.main(input)

# --- 5. Definição do Modelo: Discriminador (Discriminator) ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Entrada: num_channels x 28 x 28
            nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Tamanho do estado: 64 x 14 x 14

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Tamanho do estado: 128 x 7 x 7

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Tamanho do estado: 256 x 4 x 4
            
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Saída: 1 (Real ou Falso)
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# --- 6. Inicialização e Otimizadores ---
netG = Generator().to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

# Função de perda (Loss) e Otimizadores
criterion = nn.BCELoss() # Binary Cross-Entropy Loss

# Rótulos para o treinamento
real_label = 1.
fake_label = 0.

# Otimizadores Adam para G e D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Vetor de ruído fixo para visualização do progresso
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# --- 7. Loop de Treinamento (Training Loop) ---
print("Iniciando o Treinamento da DCGAN...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        # ----------------------------------------------------
        # (1) Treinamento do Discriminador: Maximizar log(D(x)) + log(1 - D(G(z)))
        # ----------------------------------------------------
        netD.zero_grad()
        
        # Treinamento com todas as imagens 'reais'
        real_images = data[0].to(device)
        b_size = real_images.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        output = netD(real_images)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Treinamento com todas as imagens 'falsas' geradas
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise)
        label.fill_(fake_label)
        
        # Classifica todas as imagens falsas
        output = netD(fake_images.detach()) # .detach() importante aqui!
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        # Atualiza o Discriminador
        errD = errD_real + errD_fake
        optimizerD.step()

        # ----------------------------------------------------
        # (2) Treinamento do Gerador: Maximizar log(D(G(z)))
        # ----------------------------------------------------
        netG.zero_grad()
        label.fill_(real_label) # Rótulos 'reais' para o Gerador (truque da GAN)
        
        output = netD(fake_images)
        errG = criterion(output, label)
        
        # Calcula gradientes para G
        errG.backward()
        D_G_z2 = output.mean().item()
        
        # Atualiza o Gerador
        optimizerG.step()
        
        # --- Relatório de Progresso ---
        if i % 100 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

# --- 8. Geração e Visualização das Imagens Finais ---
netG.eval()
with torch.no_grad():
    generated_images = netG(fixed_noise).cpu()

# Desnormalizar as imagens (de -1 para 1 -> 0 para 1)
generated_images = (generated_images + 1) / 2

# Plotar as imagens geradas (64 imagens no grid 8x8)
plt.figure(figsize=(10, 10))
for i in range(64):
    ax = plt.subplot(8, 8, i + 1)
    # Acessa o array numpy da imagem, remove a dimensão do canal (squeeze)
    plt.imshow(generated_images[i].squeeze(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle(f"Imagens Geradas pela DCGAN após {num_epochs} Épocas", fontsize=16)
plt.show()

print("Treinamento concluído e imagens geradas.")