## https://chatgpt.com/share/67b2e9fd-8a38-800c-b456-bec67de38831

import numpy as np
import torch

def cosine_schedule(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f_t = torch.cos(((t / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    betas = torch.clip(1 - (f_t[1:] / f_t[:-1]), 0.0001, 0.999)
    return betas

T = 1000
beta_schedule = cosine_schedule(T)

import torch.nn as nn
import torch.nn.functional as F

class DDPM_UNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, input_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)

def ddpm_sample(model, steps, input_dim):
    x_t = torch.randn((1, input_dim))  # Start with pure noise
    
    for t in reversed(range(1, steps)):
        pred_noise = model(x_t, torch.tensor([t]))  # Predict noise
        beta_t = noise_schedule[t]
        
        # Improved DDPM denoising step
        noise = torch.randn_like(x_t) if t > 1 else 0
        x_t = (x_t - beta_t * pred_noise) / torch.sqrt(1 - beta_t) + noise * torch.sqrt(beta_t)

    return x_t
