import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 超参数 ====================
batch_size = 128        # 批次大小（2D VAE可以用更大的batch）
latent_dim = 2          # 2维潜在空间
input_dim = 784         # 28*28
epochs = 30             # 训练轮次
lr = 1e-3               # 学习率

# ==================== 数据集 ====================
transform = transforms.Compose([
    transforms.ToTensor()   # 像素值0-1
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# ==================== VAE模型 ====================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),      # 784 -> 512
            nn.ReLU(),
            nn.Linear(512, 256),            # 512 -> 256
            nn.ReLU(),
            nn.Linear(256, 2*latent_dim)    # 256 -> 2*latent_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        """重参数技巧: z = μ + σ·ε"""
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z

    def forward(self, x):
        encoder_output = self.encoder(x)
        mu = encoder_output[:, :self.latent_dim]
        log_var = encoder_output[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)
        x_re = self.decoder(z)
        return x_re, mu, log_var

# ==================== 损失函数 ====================
def vae_loss(x_re, x, mu, log_var):
    """VAE损失 = 重建误差 + KL散度"""
    re_loss = nn.BCELoss(reduction='sum')(x_re, x)
    kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(log_var)**2 - log_var - 1)
    total_loss = re_loss + kl_loss
    return total_loss, re_loss, kl_loss

# ==================== 训练模型 ====================
print("="*60)
print("Training 2D VAE for Latent Space Visualization")
print("="*60)
print(f"Device: {device}")
print(f"Latent Dimension: {latent_dim}")
print(f"Training Samples: {len(train_dataset)}")
print("="*60)

model_2d = VAE(input_dim, latent_dim).to(device)
optimizer_2d = torch.optim.Adam(model_2d.parameters(), lr=lr)

model_2d.train()
for epoch in range(epochs):
    total_loss = 0.0
    total_re_loss = 0.0
    total_kl_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        x = data.view(-1, input_dim).to(device)
        x_re, mu, log_var = model_2d(x)
        loss, re_loss, kl_loss = vae_loss(x_re, x, mu, log_var)

        optimizer_2d.zero_grad()
        loss.backward()
        optimizer_2d.step()

        total_loss += loss.item()
        total_re_loss += re_loss.item()
        total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_re_loss = total_re_loss / len(train_loader.dataset)
    avg_kl_loss = total_kl_loss / len(train_loader.dataset)

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Loss: {avg_loss:.4f}, '
          f'Recon: {avg_re_loss:.4f}, '
          f'KL: {avg_kl_loss:.4f}')

print("\nTraining Finished!")

# ==================== 保存模型 ====================
torch.save({
    'model_state_dict': model_2d.state_dict(),
    'optimizer_state_dict': optimizer_2d.state_dict(),
}, 'vae_2d_model.pth')
print("Model saved to vae_2d_model.pth")

# ==================== 可视化1：潜在空间散点图 ====================
print("\nGenerating 2D latent space scatter plot...")
model_2d.eval()

all_mu = []
all_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        x = data.view(-1, input_dim).to(device)
        encoder_output = model_2d.encoder(x)
        mu = encoder_output[:, :latent_dim]
        all_mu.append(mu.cpu())
        all_labels.append(labels)

all_mu = torch.cat(all_mu).numpy()
all_labels = torch.cat(all_labels).numpy()

# 绘制散点图
plt.figure(figsize=(12, 10))
scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels,
                      cmap='tab10', alpha=0.6, s=5)
plt.colorbar(scatter, label='Digit Class')
plt.xlabel('Latent Dimension 1', fontsize=12)
plt.ylabel('Latent Dimension 2', fontsize=12)
plt.title('2D Latent Space Visualization of MNIST Digits', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('vae_latent_space_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Scatter plot saved to vae_latent_space_2d.png")

# ==================== 可视化2：潜在空间网格图 ====================
print("\nGenerating images from 2D latent space grid...")

grid_range = 3
grid_points = 20
z1 = torch.linspace(-grid_range, grid_range, grid_points)
z2 = torch.linspace(-grid_range, grid_range, grid_points)
z1_grid, z2_grid = torch.meshgrid(z1, z2, indexing='ij')
z_grid = torch.stack([z1_grid.flatten(), z2_grid.flatten()], dim=1)

with torch.no_grad():
    generated_imgs = model_2d.decoder(z_grid.to(device)).view(-1, 1, 28, 28)

# 绘制网格图像
fig, axes = plt.subplots(grid_points, grid_points, figsize=(12, 12))
for i in range(grid_points):
    for j in range(grid_points):
        idx = i * grid_points + j
        axes[i, j].imshow(generated_imgs[idx].cpu().squeeze().numpy(), cmap='gray')
        axes[i, j].axis('off')

plt.suptitle(f'Generated Digits from 2D Latent Space Grid (range: [-{grid_range}, {grid_range}])',
             fontsize=14)
plt.tight_layout()
plt.savefig('vae_latent_grid.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grid visualization saved to vae_latent_grid.png")

# ==================== 可视化3：重建效果对比 ====================
print("\nGenerating reconstruction comparison...")
with torch.no_grad():
    data, labels = next(iter(test_loader))
    x = data.view(-1, input_dim).to(device)
    x_re, _, _ = model_2d(x)

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        # 原始图像
        axes[0, i].imshow(data[i].squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # 重建图像
        axes[1, i].imshow(x_re[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.suptitle('2D VAE: Original vs Reconstructed', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_2d_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Reconstruction comparison saved to vae_2d_reconstruction.png")

print("\n" + "="*60)
print("All visualizations completed!")
print("="*60)
