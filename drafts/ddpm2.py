## https://chatgpt.com/share/67b2e9fd-8a38-800c-b456-bec67de38831

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== 1. Noise Schedule (Cosine) ========================== #
def cosine_schedule(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f_t = torch.cos(((t / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    betas = torch.clip(1 - (f_t[1:] / f_t[:-1]), 0.0001, 0.999)
    return betas

T = 1000  # Number of diffusion steps
betas = cosine_schedule(T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# ========================== 2. Forward Diffusion (Adding Noise) ========================== #
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
    return xt, noise  # Return noisy sample and actual noise

# ========================== 3. U-Net Model for Denoising ========================== #
class UNet(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=64):
        super().__init__()
        self.enc1 = nn.Conv2d(img_channels, hidden_dim, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)
        self.dec1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.dec3 = nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_embedding = t.view(-1, 1, 1, 1).repeat(1, x.shape[1], x.shape[2], x.shape[3])  # Time embedding
        x = F.relu(self.enc1(x + t_embedding))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return self.dec3(x)  # Predict noise

# ========================== 4. Training Function ========================== #
def train_step(model, x0, optimizer):
    model.train()
    t = torch.randint(1, T, (x0.shape[0],)).to(device)
    xt, true_noise = forward_diffusion(x0, t)
    pred_noise = model(xt, t)
    loss = F.mse_loss(pred_noise, true_noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ========================== 5. DDPM Sampling (Reverse Process) ========================== #
@torch.no_grad()
def sample_ddpm(model, steps, img_size):
    model.eval()
    x_t = torch.randn((1, 1, img_size, img_size)).to(device)
    for t in reversed(range(1, steps)):
        pred_noise = model(x_t, torch.tensor([t]).to(device))
        beta_t = betas[t]
        noise = torch.randn_like(x_t) if t > 1 else 0
        x_t = (x_t - beta_t * pred_noise) / torch.sqrt(1 - beta_t) + noise * torch.sqrt(beta_t)
    return x_t

# ========================== 6. DDIM Sampling (Faster) ========================== #
@torch.no_grad()
def sample_ddim(model, steps, img_size, eta=0.0):
    model.eval()
    x_t = torch.randn((1, 1, img_size, img_size)).to(device)
    tau = torch.linspace(T, 1, steps, dtype=torch.long).to(device)
    for i in range(steps - 1):
        t, t_next = tau[i], tau[i+1]
        pred_noise = model(x_t, torch.tensor([t]).to(device))
        alpha_t = alphas_cumprod[t]
        alpha_t_next = alphas_cumprod[t_next]
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        noise = torch.randn_like(x_t) if eta > 0 and t_next > 1 else 0
        x_t = torch.sqrt(alpha_t_next) * x_0_pred + torch.sqrt(1 - alpha_t_next) * noise
    return x_t

# ========================== 7. Load Dataset (MNIST or CIFAR-10) ========================== #
dataset_name = "MNIST"  # Change to "CIFAR10" if needed
transform = transforms.Compose([transforms.ToTensor()])

if dataset_name == "MNIST":
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    img_channels, img_size = 1, 28
elif dataset_name == "CIFAR10":
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    img_channels, img_size = 3, 32

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# ========================== 8. Initialize Model & Train ========================== #
model = UNet(img_channels=img_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    epoch_loss = 0
    for images, _ in trainloader:
        images = images.to(device)
        loss = train_step(model, images, optimizer)
        epoch_loss += loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(trainloader):.4f}")

# ========================== 9. Generate Images ========================== #
def generate_images(model, sampling_method="ddpm"):
    model.eval()
    if sampling_method == "ddpm":
        generated_image = sample_ddpm(model, steps=T, img_size=img_size)
    else:
        generated_image = sample_ddim(model, steps=50, img_size=img_size)
    
    generated_image = generated_image.cpu().detach().squeeze()
    if img_channels == 3:
        generated_image = generated_image.permute(1, 2, 0)  # Convert to (H, W, C)

    plt.imshow(generated_image, cmap="gray" if img_channels == 1 else None)
    plt.axis("off")
    plt.show()

# Generate image using DDPM
generate_images(model, sampling_method="ddpm")

# Generate image using faster DDIM
generate_images(model, sampling_method="ddim")
