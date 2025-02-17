## https://chatgpt.com/share/67b2e9fd-8a38-800c-b456-bec67de38831

import torch
import torch.nn as nn

def forward_diffusion(x0, t, noise_schedule):
    """Applies the forward diffusion process to add noise."""
    noise = torch.randn_like(x0)  # Gaussian noise
    alpha_t = torch.prod(1 - noise_schedule[:t], dim=0)  # Cumulative noise schedule
    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise  # Noisy sample
    return xt, noise

class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # Output has same shape as input noise
        )

    def forward(self, x_t, t):
        return self.net(x_t)

def train_step(model, x0, noise_schedule, optimizer, t):
    # t = torch.randint(1, len(noise_schedule), (x0.shape[0],))  # Random time step
    xt, true_noise = forward_diffusion(x0, t, noise_schedule)  # Noisy sample
    pred_noise = model(xt, t)  # Model predicts noise
    
    loss = torch.nn.functional.mse_loss(pred_noise, true_noise)  # MSE loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def sample(model, steps, input_dim):
    x_t = torch.randn((1, input_dim))  # Start with random noise
    
    for t in reversed(range(1, steps)):
        pred_noise = model(x_t, torch.tensor([t]))  # Predict noise
        beta_t = noise_schedule[t]  # Noise variance
        x_t = (x_t - beta_t * pred_noise) / torch.sqrt(1 - beta_t)  # Denoise
    
    return x_t

if __name__ == "__main__":
    ## noise schedule
    T = 1000  # Total time steps
    beta_min, beta_max = 0.0001, 0.02  # Define range of noise
    noise_schedule = torch.linspace(beta_min, beta_max, T)  # Linear increase

    input_dim=5
    x0 = torch.arange(input_dim).to(torch.float)
    
    model = SimpleDenoiser(input_dim)

    optimizer = torch.optim.Adam(model.parameters())

    ## epochs
    for _ in range(100):
        ## single training round
        for i in range(10):
            train_step(model, x0, noise_schedule, optimizer, i)
    
    t = 5
    x5 = forward_diffusion(x0, t, noise_schedule)
    print(x0)
    print(sample(model, 5, input_dim))