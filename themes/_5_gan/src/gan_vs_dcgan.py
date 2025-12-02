import os
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
z_dim = 100

EPOCHS_GAN = 200
EPOCHS_DCGAN = 100

os.makedirs("results/gan", exist_ok=True)
os.makedirs("results/dcgan", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

loader = DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

class GAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.fc(z)
        return img.view(-1, 1, 28, 28)

class GAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.fc(img.view(img.size(0), -1))

class DCGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z.view(-1, z_dim, 1, 1))

class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.disc(img)

def train(model_name, Generator, Discriminator, save_dir, total_epochs):
    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(25, z_dim, device=device)

    for epoch in range(1, total_epochs + 1): # Usa total_epochs
        for real, _ in loader:
            real = real.to(device)
            bs = real.size(0)

            # --------- Treina Discriminador ----------
            noise = torch.randn(bs, z_dim, device=device)
            fake = G(noise)

            # Labels reais (1) e falsas (0)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            loss_D = (
                criterion(D(real), real_labels) +
                criterion(D(fake.detach()), fake_labels)
            ) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --------- Treina Gerador ----------
            noise = torch.randn(bs, z_dim, device=device)
            fake = G(noise)

            # Gerador quer que o discriminador aceite o fake como real (labels 1)
            loss_G = criterion(D(fake), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        # salva imagem do progresso
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            grid = samples.view(25, 1, 28, 28)

            fig, axes = plt.subplots(5, 5, figsize=(5, 5))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(grid[i][0], cmap="gray")
                ax.axis("off")
            # Usa f-string para garantir que o número da época está correto
            plt.savefig(f"{save_dir}/epoch_{epoch}.png") 
            plt.close()

        print(f"[{model_name}] Epoch {epoch}/{total_epochs}") # Usa total_epochs

    return G

print("GAN...")
G_gan = train("GAN", GAN_Generator, GAN_Discriminator, "results/gan", EPOCHS_GAN)
print("DCGAN...")
G_dcgan = train("DCGAN", DCGAN_Generator, DCGAN_Discriminator, "results/dcgan", EPOCHS_DCGAN)

def load_img(path):
    return Image.open(path)

gan_img = load_img(f"results/gan/epoch_{EPOCHS_GAN}.png")
dcgan_img = load_img(f"results/dcgan/epoch_{EPOCHS_DCGAN}.png")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(gan_img, cmap="gray")
plt.title(f"GAN após {EPOCHS_GAN} épocas") 
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(dcgan_img, cmap="gray")
plt.title(f"DCGAN após {EPOCHS_DCGAN} épocas") 
plt.axis("off")

plt.tight_layout()
plt.savefig("comparacao_final.png")
plt.show()