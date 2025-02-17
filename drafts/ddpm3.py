import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ========================== 1. Self-Attention Module ========================== #
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1)  # (B, C, H*W)
        k = self.key(x).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        v = self.value(x).view(B, C, -1)  # (B, C, H*W)
        
        attn = torch.bmm(q, k) / (C ** 0.5)  # Scaled dot-product
        attn = F.softmax(attn, dim=-1)  # (B, C, C)
        out = torch.bmm(v, attn).view(B, C, H, W)  # (B, C, H, W)

        return self.gamma * out + x  # Skip connection

# ========================== 2. Class-Conditional Attention U-Net ========================== #
class ClassConditionalAttentionUNet(nn.Module):
    def __init__(self, img_channels=1, num_classes=10, hidden_dim=64):
        super().__init__()
        self.embed_class = nn.Embedding(num_classes, hidden_dim)  # Class embedding

        # Encoding layers
        self.enc1 = nn.Conv2d(img_channels, hidden_dim, kernel_size=3, padding=1)
        self.attn1 = SelfAttention(hidden_dim)
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.attn2 = SelfAttention(hidden_dim * 2)
        self.enc3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)
        self.attn3 = SelfAttention(hidden_dim * 4)

        # Decoding layers
        self.dec1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.dec3 = nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t, labels):
        class_embed = self.embed_class(labels).view(-1, 1, 1, 1)  # Embed class label
        t_embedding = t.view(-1, 1, 1, 1).repeat(1, x.shape[1], x.shape[2], x.shape[3])

        # Forward through encoder with attention
        x = F.relu(self.attn1(self.enc1(x + class_embed + t_embedding)))
        x = F.relu(self.attn2(self.enc2(x)))
        x = F.relu(self.attn3(self.enc3(x)))

        # Forward through decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return self.dec3(x)  # Predict noise

# ========================== 3. Forward (Noising) Process ========================== #
def forward_diffusion(x, t, noise):
    alpha_t = (1 - 0.02 * t / 1000).to(x.device)  # Noise schedule
    return alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise

# ========================== 4. Training Pipeline ========================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
batch_size = 128
lr = 1e-3
timesteps = 1000

# Load dataset (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model & optimizer
model = ClassConditionalAttentionUNet(img_channels=1, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    for i, (real, labels) in enumerate(dataloader):
        real, labels = real.to(device), labels.to(device)
        noise = torch.randn_like(real)  # Random Gaussian noise
        t = torch.randint(1, timesteps, (real.shape[0],)).to(device)  # Random timestep
        noisy_x = forward_diffusion(real, t, noise)  # Add noise

        pred_noise = model(noisy_x, t, labels)  # Predict noise
        loss = loss_fn(pred_noise, noise)  # MSE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

# ========================== 5. Sampling (Image Generation) ========================== #
@torch.no_grad()
def sample(num_samples=10, class_label=3):
    model.eval()
    samples = torch.randn(num_samples, 1, 28, 28).to(device)  # Start from noise
    labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)

    for t in reversed(range(1, timesteps)):  # Reverse diffusion
        noise = torch.randn_like(samples) if t > 1 else torch.zeros_like(samples)
        pred_noise = model(samples, torch.full((num_samples,), t, dtype=torch.long).to(device), labels)
        alpha_t = (1 - 0.02 * t / 1000).to(device)
        samples = (samples - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt() + (1 - alpha_t).sqrt() * noise

    return samples.cpu()

# Generate & plot images
samples = sample(10, class_label=3)  # Generate 10 images of digit "3"
fig, axes = plt.subplots(1, 10, figsize=(10, 2))
for i, img in enumerate(samples):
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].axis("off")
plt.show()
