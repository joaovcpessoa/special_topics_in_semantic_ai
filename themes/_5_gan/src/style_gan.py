import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Function

# ==============================================================================
# 1. Utilitários de Baixo Nível (PyTorch)
# ==============================================================================

# Implementação do Desfoque 2D com gradiente customizado (equivalente a blur2d do TF)
# Usado para suavizar as transições no Progressive Growing e no downscale/upscale.
class Blur2d(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32)
        if kernel.ndim == 1:
            kernel = kernel[:, np.newaxis] * kernel[np.newaxis, :]
        
        if normalize:
            kernel /= np.sum(kernel)
        
        if flip:
            kernel = kernel[::-1, ::-1]
            
        self.register_buffer('kernel', torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0))
        self.stride = stride
        self.padding = (kernel.shape[0] // 2, kernel.shape[1] // 2)

    def forward(self, x):
        # x shape: [N, C, H, W]
        C = x.shape[1]
        # Aplica o kernel a cada canal (grupos = C)
        return F.conv2d(x, self.kernel.repeat(C, 1, 1, 1), stride=self.stride, padding=self.padding, groups=C)

# Implementação do Upscale 2D (equivalente a upscale2d do TF)
def upscale2d(x, factor=2):
    if factor == 1:
        return x
    # Usa F.interpolate para redimensionamento
    return F.interpolate(x, scale_factor=factor, mode='nearest')

# Implementação do Downscale 2D (equivalente a downscale2d do TF)
def downscale2d(x, factor=2):
    if factor == 1:
        return x
    # Para o StyleGAN, o downscale 2x2 é feito com um blur antes do subsampling
    if factor == 2:
        # O blur é aplicado implicitamente no conv2d_downscale2d, mas para o downscale puro
        # podemos usar a média (avg_pool)
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)
    else:
        raise NotImplementedError("Downscale por fator > 2 não implementado para esta conversão.")

# Implementação do Equalized Learning Rate (WScale)
class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gain=np.sqrt(2), lrmul=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Inicialização He
        fan_in = in_channels * kernel_size * kernel_size
        self.he_std = gain / np.sqrt(fan_in)
        
        # Coeficiente de tempo de execução (runtime_coef)
        self.runtime_coef = self.he_std * lrmul
        
        # Inicialização do peso (sem o runtime_coef)
        init_std = 1.0 / lrmul
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * init_std)
        
        self.lrmul = lrmul

    def forward(self, x):
        # Multiplica o peso pelo coeficiente de tempo de execução (Equalized LR)
        w = self.weight * self.runtime_coef
        return F.conv2d(x, w, stride=self.stride, padding=self.padding)

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, gain=np.sqrt(2), lrmul=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Inicialização He
        fan_in = in_features
        self.he_std = gain / np.sqrt(fan_in)
        
        # Coeficiente de tempo de execução (runtime_coef)
        self.runtime_coef = self.he_std * lrmul
        
        # Inicialização do peso (sem o runtime_coef)
        init_std = 1.0 / lrmul
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * init_std)
        
        self.lrmul = lrmul

    def forward(self, x):
        # Multiplica o peso pelo coeficiente de tempo de execução (Equalized LR)
        w = self.weight * self.runtime_coef
        return F.linear(x, w)

# Aplica Bias (com lrmul)
class ApplyBias(nn.Module):
    def __init__(self, num_features, lrmul=1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.lrmul = lrmul

    def forward(self, x):
        b = self.bias * self.lrmul
        if x.dim() == 2: # FC layer output
            return x + b
        elif x.dim() == 4: # Conv layer output (NCHW)
            return x + b.view(1, -1, 1, 1)
        else:
            raise NotImplementedError("Dimensão de entrada não suportada para ApplyBias.")

# Leaky ReLU (Função auxiliar para chamadas funcionais)
def functional_leaky_relu(x, alpha=0.2):
    return F.leaky_relu(x, negative_slope=alpha)

# Pixelwise feature vector normalization
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x: [N, C, H, W]
        # Calcula a média do quadrado ao longo do canal (dim=1)
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + self.epsilon)

# Instance normalization
class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x: [N, C, H, W]
        # Subtrai a média e divide pelo desvio padrão ao longo de H e W
        x = x - torch.mean(x, dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(torch.mean(x.pow(2), dim=[2, 3], keepdim=True) + self.epsilon)
        return x

# Style modulation
class StyleMod(nn.Module):
    def __init__(self, num_features, dlatent_size, lrmul=1):
        super().__init__()
        self.num_features = num_features
        # Camada FC para mapear dlatent para o estilo (2 * num_features: scale e bias)
        self.style_fc = EqualizedLinear(dlatent_size, num_features * 2, gain=1, lrmul=lrmul)
        self.bias = nn.Parameter(torch.zeros(num_features * 2))
        self.lrmul = lrmul

    def forward(self, x, dlatent):
        # dlatent: [N, dlatent_size]
        style = self.style_fc(dlatent) + self.bias * self.lrmul
        # style: [N, 2 * num_features] -> [N, 2, num_features, 1, 1]
        style = style.view(-1, 2, self.num_features, 1, 1)
        
        # Modulação: x * (scale + 1) + bias
        return x * (style[:, 0] + 1) + style[:, 1]

# Camada de ruído
class ApplyNoise(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # O StyleGAN original usa um tensor de ruído fixo por camada, mas com um peso treinável
        self.weight = nn.Parameter(torch.zeros(1)) # Peso treinável para o ruído
        self.register_buffer('noise', None) # O ruído real será gerado no forward

    def forward(self, x, noise=None):
        # x: [N, C, H, W]
        if noise is None:
            # Gera ruído aleatório com a mesma forma espacial de x
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        
        # Adiciona o ruído escalado
        return x + noise * self.weight

# ==============================================================================
# 2. Blocos de Rede (PyTorch)
# ==============================================================================

# Bloco Convolucional Básico (Conv + Bias + Act)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2), lrmul=1, activation='lrelu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding, gain=gain, lrmul=lrmul)
        self.bias = ApplyBias(out_channels, lrmul=lrmul)
        self.act = nn.LeakyReLU(0.2) if activation == 'lrelu' else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bias(x)
        x = self.act(x)
        return x

# Bloco do Gerador (StyleGAN)
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dlatent_size, resolution, blur_filter=[1, 2, 1], lrmul=1):
        super().__init__()
        self.res = resolution
        self.blur = Blur2d(blur_filter)
        
        # Conv 0 (Upscale + Conv + Blur)
        self.conv0 = nn.Sequential(
            # O upscale é feito no forward
            EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1, gain=np.sqrt(2), lrmul=lrmul),
            self.blur
        )
        self.noise0 = ApplyNoise(out_channels)
        self.style0 = StyleMod(out_channels, dlatent_size, lrmul=lrmul)
        self.bias0 = ApplyBias(out_channels, lrmul=lrmul)
        
        # Conv 1
        self.conv1 = ConvBlock(out_channels, out_channels, kernel_size=3, lrmul=lrmul)
        self.noise1 = ApplyNoise(out_channels)
        self.style1 = StyleMod(out_channels, dlatent_size, lrmul=lrmul)
        self.bias1 = ApplyBias(out_channels, lrmul=lrmul)

    def forward(self, x, dlatents):
        # dlatents: [N, 2, dlatent_size] (para 2 camadas)
        
        # Conv 0 (Upscale + Conv + Blur)
        x = upscale2d(x, factor=2)
        x = self.conv0(x)
        x = self.noise0(x)
        x = self.style0(x, dlatents[:, 0])
        x = functional_leaky_relu(self.bias0(x))
        
        # Conv 1
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.style1(x, dlatents[:, 1])
        x = functional_leaky_relu(self.bias1(x))
        
        return x

# Bloco do Discriminador (StyleGAN)
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, blur_filter=[1, 2, 1], lrmul=1):
        super().__init__()
        self.res = resolution
        self.blur = Blur2d(blur_filter)
        
        # Conv 0
        self.conv0 = ConvBlock(in_channels, in_channels, kernel_size=3, lrmul=lrmul)
        
        # Conv 1 (Conv + Blur + Downscale)
        self.conv1 = nn.Sequential(
            EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1, gain=np.sqrt(2), lrmul=lrmul),
            ApplyBias(out_channels, lrmul=lrmul),
            nn.LeakyReLU(0.2),
            self.blur,
            nn.AvgPool2d(kernel_size=2, stride=2) # Downscale
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        return x

# ==============================================================================
# 3. Redes G e D (PyTorch)
# ==============================================================================

class Generator(nn.Module):
    def __init__(self, resolution, z_dim=512, dlatent_size=512, num_channels=3, fmap_base=8192, fmap_decay=1.0, fmap_max=512, lrmul=1):
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        self.z_dim = z_dim
        self.dlatent_size = dlatent_size
        self.num_channels = num_channels
        
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        # Mapeamento (8 camadas FC)
        layers = []
        for i in range(8):
            layers.append(EqualizedLinear(dlatent_size, dlatent_size, lrmul=lrmul))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)
        
        # Camadas de Síntese
        self.synthesis = nn.ModuleList()
        
        # 4x4 (Input)
        self.const_input = nn.Parameter(torch.randn(1, nf(1), 4, 4))
        
        # 4x4 (Conv)
        self.synthesis.append(nn.ModuleDict({
            'noise0': ApplyNoise(nf(1)),
            'style0': StyleMod(nf(1), dlatent_size, lrmul=lrmul),
            'bias0': ApplyBias(nf(1), lrmul=lrmul),
            'conv1': ConvBlock(nf(1), nf(1), kernel_size=3, lrmul=lrmul),
            'noise1': ApplyNoise(nf(1)),
            'style1': StyleMod(nf(1), dlatent_size, lrmul=lrmul),
            'bias1': ApplyBias(nf(1), lrmul=lrmul),
            'torgb': ConvBlock(nf(1), num_channels, kernel_size=1, gain=1, lrmul=lrmul, activation='linear')
        }))
        
        # Blocos progressivos (8x8 até a resolução final)
        for res in range(3, self.resolution_log2 + 1):
            in_ch = nf(res - 1)
            out_ch = nf(res - 1)
            
            self.synthesis.append(nn.ModuleDict({
                'block': GBlock(in_ch, out_ch, dlatent_size, 2**res, lrmul=lrmul),
                'torgb': ConvBlock(out_ch, num_channels, kernel_size=1, gain=1, lrmul=lrmul, activation='linear')
            }))

    def forward(self, z, lod=0.0):
        # 1. Mapeamento (z -> w)
        w = self.mapping(z)
        
        # 2. Síntese
        x = self.const_input.repeat(z.shape[0], 1, 1, 1)
        
        # 4x4 (Bloco inicial)
        block_4x4 = self.synthesis[0]
        
        # Conv 0 (Input)
        x = block_4x4['noise0'](x)
        x = block_4x4['style0'](x, w)
        x = functional_leaky_relu(block_4x4['bias0'](x))
        
        # Conv 1
        x = block_4x4['conv1'](x)
        x = block_4x4['noise1'](x)
        x = block_4x4['style1'](x, w)
        x = functional_leaky_relu(block_4x4['bias1'](x))
        
        # ToRGB inicial (para o fade-in)
        img = block_4x4['torgb'](x)
        
        # Blocos progressivos
        for res_idx in range(1, len(self.synthesis)):
            res = res_idx + 2 # 8, 16, 32, ...
            
            # Se o LOD for maior que o LOD atual, faz o fade-in
            if lod > self.resolution_log2 - res:
                # Bloco atual
                block = self.synthesis[res_idx]
                x = block['block'](x, w.unsqueeze(1).repeat(1, 2, 1)) # Simplificação: usa o mesmo w para as 2 camadas
                
                # Novo ToRGB
                new_img = block['torgb'](x)
                
                # Interpolação (Fade-in)
                alpha = lod - (self.resolution_log2 - res)
                img = upscale2d(img, factor=2)
                img = torch.lerp(img, new_img, alpha)
            else:
                # Apenas upscale da imagem anterior (se estiver no nível de detalhe mais baixo)
                img = upscale2d(img, factor=2)
        
        return img

class Discriminator(nn.Module):
    def __init__(self, resolution, num_channels=3, fmap_base=8192, fmap_decay=1.0, fmap_max=512, lrmul=1):
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        self.num_channels = num_channels
        
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.blocks = nn.ModuleList()
        
        # Blocos progressivos (da resolução final até 8x8)
        for res in range(self.resolution_log2, 2, -1):
            in_ch = nf(res - 1)
            out_ch = nf(res - 2)
            
            # FromRGB
            self.blocks.append(nn.ModuleDict({
                'fromrgb': ConvBlock(num_channels, in_ch, kernel_size=1, lrmul=lrmul),
                'block': DBlock(in_ch, out_ch, 2**res, lrmul=lrmul)
            }))
            
        # 4x4 (Bloco final)
        self.final_block = nn.ModuleDict({
            'fromrgb': ConvBlock(num_channels, nf(1), kernel_size=1, lrmul=lrmul),
            'minibatch_stddev': nn.Identity(), # Simplificação: não implementaremos o Minibatch Stddev
            'conv': ConvBlock(nf(1), nf(1), kernel_size=3, lrmul=lrmul),
            'dense0': EqualizedLinear(nf(1) * 4 * 4, nf(0), lrmul=lrmul),
            'dense1': EqualizedLinear(nf(0), 1, gain=1, lrmul=lrmul) # Output score
        })

    def forward(self, img, lod=0.0):
        x = None
        
        # Blocos progressivos
        for res_idx in range(len(self.blocks) - 1, -1, -1):
            res = self.resolution_log2 - res_idx # 1024, 512, 256, ...
            block = self.blocks[res_idx]
            
            # Se o LOD for maior que o LOD atual, faz o fade-in
            if lod > self.resolution_log2 - res:
                # FromRGB do nível atual
                new_x = block['fromrgb'](img)
                
                # Se x já existe (nível anterior), faz a interpolação
                if x is not None:
                    # Downscale da imagem anterior
                    img_down = downscale2d(img, factor=2)
                    x_prev = block['fromrgb'](img_down)
                    
                    # Interpolação (Fade-in)
                    alpha = lod - (self.resolution_log2 - res)
                    x = torch.lerp(x_prev, new_x, alpha)
                else:
                    x = new_x
                
                # Bloco de convolução
                x = block['block'](x)
            else:
                # Apenas downscale da imagem anterior (se estiver no nível de detalhe mais baixo)
                img = downscale2d(img, factor=2)
        
        # Bloco final (4x4)
        final_block = self.final_block
        
        # FromRGB final
        if x is None:
            x = final_block['fromrgb'](img)
        
        # Minibatch Stddev (simplificado)
        x = final_block['minibatch_stddev'](x)
        
        # Conv
        x = final_block['conv'](x)
        
        # Flatten
        x = x.view(x.shape[0], -1)
        
        # Dense 0
        x = functional_leaky_relu(final_block['dense0'](x))
        
        # Dense 1 (Output score)
        score = final_block['dense1'](x)
        
        return score

# ==============================================================================
# 4. Funções de Perda (PyTorch)
# ==============================================================================

# Perda do Gerador (Non-saturating logistic loss)
def G_loss_logistic_nonsaturating(G, D, z, labels):
    fake_images = G(z)
    fake_scores = D(fake_images, lod=0.0) # Assume lod=0.0 para o cálculo da perda
    # -log(logistic(fake_scores)) = softplus(-fake_scores)
    loss = F.softplus(-fake_scores).mean()
    return loss

# Perda do Discriminador (Logistic loss + R1 regularization)
def D_loss_logistic_simplegp(G, D, reals, z, labels, r1_gamma=10.0):
    # 1. Perda real
    real_scores = D(reals, lod=0.0)
    loss_real = F.softplus(-real_scores).mean() # -log(logistic(real_scores))
    
    # 2. Perda falsa
    with torch.no_grad():
        fake_images = G(z).detach()
    fake_scores = D(fake_images, lod=0.0)
    loss_fake = F.softplus(fake_scores).mean() # -log(1 - logistic(fake_scores))
    
    loss = loss_real + loss_fake
    
    # 3. Penalidade R1 (Gradient Penalty)
    if r1_gamma > 0.0:
        reals.requires_grad = True
        real_scores_r1 = D(reals, lod=0.0)
        
        # Calcula o gradiente de real_scores_r1 em relação a reals
        grad_outputs = torch.ones_like(real_scores_r1)
        gradients = torch.autograd.grad(
            outputs=real_scores_r1,
            inputs=reals,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # R1 penalty: ||grad(D(x))||^2
        r1_penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
        loss += r1_penalty * (r1_gamma * 0.5)
        
        reals.requires_grad = False # Volta ao estado original
        
    return loss

# ==============================================================================
# 5. Loop de Treinamento (PyTorch)
# ==============================================================================

def training_loop_pytorch(
    G, D, Gs,
    dataset_loader,
    total_kimg=25000,
    sched_args={},
    G_opt_args={'lr': 0.001, 'betas': (0.0, 0.99), 'eps': 1e-8},
    D_opt_args={'lr': 0.001, 'betas': (0.0, 0.99), 'eps': 1e-8},
    r1_gamma=10.0,
    z_dim=512,
    device='cpu'
):
    # Otimizadores
    G_opt = torch.optim.Adam(G.parameters(), **G_opt_args)
    D_opt = torch.optim.Adam(D.parameters(), **D_opt_args)
    
    # Parâmetros do agendamento (simplificado)
    resolution_log2 = G.resolution_log2
    lod_initial_resolution = sched_args.get('lod_initial_resolution', 4)
    lod_training_kimg = sched_args.get('lod_training_kimg', 600)
    lod_transition_kimg = sched_args.get('lod_transition_kimg', 600)
    
    # Loop de treinamento (simplificado para demonstração)
    cur_nimg = 0
    max_nimg = total_kimg * 1000
    
    data_iter = iter(dataset_loader)
    
    while cur_nimg < max_nimg:
        # 1. Agendamento (LOD e LR)
        kimg = cur_nimg / 1000.0
        
        # Cálculo do LOD (simplificado do training_schedule original)
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = kimg - phase_idx * phase_dur
        
        lod = resolution_log2 - np.floor(np.log2(lod_initial_resolution)) - phase_idx
        if lod_transition_kimg > 0:
            lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        lod = max(lod, 0.0)
        
        # 2. Obter minibatch
        try:
            reals, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset_loader)
            reals, labels = next(data_iter)
            
        reals = reals.to(device)
        
        # 3. Treinar Discriminador
        D_opt.zero_grad()
        
        # Gerar ruído
        z = torch.randn(reals.shape[0], z_dim, device=device)
        
        # Calcular perda D
        D_loss = D_loss_logistic_simplegp(G, D, reals, z, labels, r1_gamma=r1_gamma)
        D_loss.backward()
        D_opt.step()
        
        # 4. Treinar Gerador
        G_opt.zero_grad()
        
        # Gerar novo ruído
        z = torch.randn(reals.shape[0], z_dim, device=device)
        
        # Calcular perda G
        G_loss = G_loss_logistic_nonsaturating(G, D, z, labels)
        G_loss.backward()
        G_opt.step()
        
        # 5. Atualizar Gs (Exponential Moving Average - EMA)
        # Gs_beta = 0.5 ** (minibatch_size / (G_smoothing_kimg * 1000.0))
        # Simplificação: Usaremos um beta fixo para demonstração
        Gs_beta = 0.999
        with torch.no_grad():
            for g_param, gs_param in zip(G.parameters(), Gs.parameters()):
                gs_param.data.mul_(Gs_beta).add_(g_param.data, alpha=1 - Gs_beta)
        
        cur_nimg += reals.shape[0]
        
        if cur_nimg % 1000 == 0:
            print(f"Kimg: {kimg:.2f}, LOD: {lod:.2f}, D_Loss: {D_loss.item():.4f}, G_Loss: {G_loss.item():.4f}")

# ==============================================================================
# 6. Exemplo de Uso (Setup e Execução)
# ==============================================================================

def main():
    # Configuração
    resolution = 1024
    z_dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Inicializar Redes
    G = Generator(resolution=resolution, z_dim=z_dim).to(device)
    D = Discriminator(resolution=resolution).to(device)
    Gs = Generator(resolution=resolution, z_dim=z_dim).to(device)
    Gs.load_state_dict(G.state_dict()) # Inicializa Gs com os pesos de G
    
    # 2. Configurar um DataLoader de Exemplo (Substitua pelo seu dataset real)
    # Criando um dataset dummy para simular o treinamento
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, resolution=1024):
            self.size = size
            self.resolution = resolution
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            # Imagens aleatórias (NCHW, range [-1, 1])
            image = torch.randn(3, self.resolution, self.resolution) * 2 - 1
            labels = torch.zeros(1) # Sem labels para simplificar
            return image, labels

    dummy_dataset = DummyDataset(resolution=resolution)
    # Minibatch size de 4 (como no exemplo do train.py para 1 GPU)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=True)
    
    print(f"Iniciando treinamento em {device}...")
    print(f"Resolução: {resolution}x{resolution}, Z-dim: {z_dim}")
    
    # 3. Iniciar o Loop de Treinamento (Apenas 100 kimg para demonstração)
    training_loop_pytorch(
        G, D, Gs,
        dataset_loader=dummy_loader,
        total_kimg=100, # Reduzido para 100 kimg para demonstração
        device=device
    )
    
    print("Treinamento de demonstração concluído.")
    
    # Exemplo de geração de imagem
    with torch.no_grad():
        z_test = torch.randn(1, z_dim, device=device)
        # LOD 0.0 (resolução máxima)
        fake_image = Gs(z_test, lod=0.0).cpu().squeeze(0) 
        print(f"Imagem gerada (Gs) com shape: {fake_image.shape}")
        
        # Salvar a imagem (necessita de conversão de tensor para imagem)
        # Exemplo: (fake_image + 1) / 2 para [0, 1] e permute(1, 2, 0) para HWC
        # ... (código para salvar imagem omitido para manter o foco na estrutura)

if __name__ == '__main__':
    main()
