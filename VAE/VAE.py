import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameter
batch_size = 16         # 批次大小
latent_dim = 20         # 隐向量维度(可调整)
input_dim = 784         # 28*28
epochs = 50             # 训练轮次
lr = 1e-3               # 学习率

transform = transforms.Compose([
    transforms.ToTensor()   # 像素值0-1
])

# dataset
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# VAE 主体
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
            nn.Linear(256, 2*latent_dim)    # 256 -> 2*latent_dim（移除Sigmoid，mu和log_var需要任意实数值）
        )

        # input: z
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    # 重参数技巧: z = μ + σ·ε
    def reparameterize(self, mu, log_var):
        # 这里用log_var（对数方差），避免直接输出方差导致非负问题
        sigma = torch.exp(0.5 * log_var)    # σ
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z
    
    def forward(self, x):
        # x: (batch_size, imput_dim)
        encoder_output = self.encoder(x)                # (batch_size, 2*latent_dim)
        mu = encoder_output[:, :self.latent_dim]        # (batch_size, latent_dim)
        log_var = encoder_output[:, self.latent_dim:]   # (batch_size, latent_dim)
        z = self.reparameterize(mu, log_var)            # (batch_size, latent_dim)

        x_re = self.decoder(z)

        return x_re, mu, log_var                        # 返回mu和log_var用于计算损失
    
model = VAE(input_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# loss = -ELBO = 重建误差 + KL散度
def vae_loss(x_re, x, mu, log_var):
    re_loss = nn.BCELoss(reduction='sum')(x_re, x)

    # KL = 0.5 * sum(μ² + σ² - logσ² - 1)，这里用log_var（logσ²）
    kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(log_var)**2 - log_var - 1)

    total_loss = re_loss + kl_loss

    return total_loss, re_loss, kl_loss

# 开始训练
model.train()

for epoch in range(epochs):
    total_loss = 0.0
    total_re_loss = 0.0
    total_kl_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        x = data.view(-1, input_dim).to(device)
        x_re, mu, log_var = model(x)
        loss, re_loss, kl_loss = vae_loss(x_re, x, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_re_loss += re_loss.item()
        total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_re_loss = total_re_loss / len(train_loader.dataset)
    avg_kl_loss = total_kl_loss / len(train_loader.dataset)

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Avg Loss: {avg_loss:.4f}, '
          f'Avg Recon Loss: {avg_re_loss:.4f}, '
          f'Avg KL Loss: {avg_kl_loss:.4f}')

print("Finished!")

# 保存模型
torch.save({
    'encoder': model.encoder.state_dict(),
    'decoder': model.decoder.state_dict(),
    'optimizer': optimizer.state_dict()
}, 'vae_model.pth')
print("Model saved to vae_model.pth")

# 可视化：原始图像 vs 重建图像
model.eval()
with torch.no_grad():
    # 获取一个batch的测试图像
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    data, _ = next(iter(test_loader))
    x = data.view(-1, input_dim).to(device)
    x_re, _, _ = model(x)

    # 显示原始图像和重建图像对比
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

    plt.suptitle('Original vs Reconstructed Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Reconstruction comparison saved to vae_reconstruction.png")

# 生成新样本
with torch.no_grad():
    # 从潜在空间随机采样生成新图像
    z = torch.randn(20, latent_dim).to(device)
    generated_imgs = model.decoder(z).view(-1, 1, 28, 28)

    plt.figure(figsize=(15, 3))
    for i in range(20):
        plt.subplot(2, 10, i+1)
        plt.imshow(generated_imgs[i].cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle('Generated MNIST Digits from Random Latent Vectors', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_generated.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Generated images saved to vae_generated.png")


# ================= 可选：2D潜在空间可视化（需要latent_dim=2）=================
print("\n" + "="*60)
print("Training 2D VAE for latent space visualization...")
print("="*60)

# 创建2D潜在空间的VAE
latent_dim_2d = 2
model_2d = VAE(input_dim, latent_dim_2d).to(device)
optimizer_2d = torch.optim.Adam(model_2d.parameters(), lr=lr)

# 训练2D VAE（较少的epoch）
epochs_2d = 20
model_2d.train()

for epoch in range(epochs_2d):
    total_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        x = data.view(-1, input_dim).to(device)
        x_re, mu, log_var = model_2d(x)
        loss, re_loss, kl_loss = vae_loss(x_re, x, mu, log_var)

        optimizer_2d.zero_grad()
        loss.backward()
        optimizer_2d.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f'2D VAE - Epoch [{epoch+1}/{epochs_2d}], Avg Loss: {avg_loss:.4f}')

# 可视化2D潜在空间
print("\nGenerating 2D latent space visualization...")
model_2d.eval()

# 收集测试数据的潜在表示
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

all_mu = []
all_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        x = data.view(-1, input_dim).to(device)
        encoder_output = model_2d.encoder(x)
        mu = encoder_output[:, :latent_dim_2d]
        all_mu.append(mu.cpu())
        all_labels.append(labels)

all_mu = torch.cat(all_mu).numpy()
all_labels = torch.cat(all_labels).numpy()

# 绘制2D潜在空间散点图
plt.figure(figsize=(12, 10))
scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=5)
plt.colorbar(scatter, label='Digit Class')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('2D Latent Space Visualization of MNIST Digits')
plt.grid(True, alpha=0.3)
plt.savefig('vae_latent_space_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print("2D latent space visualization saved to vae_latent_space_2d.png")

# 在2D潜在空间网格上生成图像
print("\nGenerating images from 2D latent space grid...")
model_2d.eval()

# 创建一个网格的潜在向量
grid_range = 3
grid_points = 20
z1 = torch.linspace(-grid_range, grid_range, grid_points)
z2 = torch.linspace(-grid_range, grid_range, grid_points)
z1_grid, z2_grid = torch.meshgrid(z1, z2)
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

plt.suptitle(f'Generated Digits from 2D Latent Space Grid (range: [-{grid_range}, {grid_range}])', fontsize=14)
plt.tight_layout()
plt.savefig('vae_latent_grid.png', dpi=150, bbox_inches='tight')
plt.show()
print("Latent space grid visualization saved to vae_latent_grid.png")