import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    """VAE 模型定义（与训练时保持一致）"""
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2*latent_dim)
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


def load_model(model_path, latent_dim=20):
    """加载保存的 VAE 模型"""
    model = VAE(input_dim=784, latent_dim=latent_dim).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # 支持两种保存格式
    if 'model_state_dict' in checkpoint:
        # VAE_2d.py 的保存格式
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {model_path}")
        if 'optimizer_state_dict' in checkpoint:
            print(f"包含优化器状态")
    elif 'encoder' in checkpoint:
        # VAE.py 的保存格式
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        print(f"加载模型: {model_path}")
        if 'optimizer' in checkpoint:
            print(f"包含优化器状态")
    else:
        model.load_state_dict(checkpoint)
        print(f"加载模型: {model_path}")

    model.eval()
    return model


def generate_random_images(model, num_images=20, save_path='vae_generated.png'):
    """从潜在空间随机采样生成图片"""
    with torch.no_grad():
        # 从标准正态分布 N(0,1) 随机采样
        z = torch.randn(num_images, model.latent_dim).to(device)
        generated_imgs = model.decoder(z).view(-1, 1, 28, 28)

        # 绘制生成的图片
        n_rows = (num_images + 9) // 10
        plt.figure(figsize=(15, 1.5 * n_rows))
        for i in range(num_images):
            plt.subplot(n_rows, 10, i + 1)
            plt.imshow(generated_imgs[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')

        plt.suptitle(f'Generated MNIST Digits (Random Sampling, latent_dim={model.latent_dim})',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"已生成 {num_images} 张图片，保存到: {save_path}")

    return generated_imgs


def generate_latent_grid(model, grid_range=3, grid_points=15, save_path='vae_latent_grid.png'):
    """生成潜在空间网格图（仅适用于 2D 潜在空间）"""
    if model.latent_dim != 2:
        print(f"⚠ 潜在空间维度为 {model.latent_dim}，无法生成 2D 网格图")
        return

    with torch.no_grad():
        # 创建潜在空间的网格点
        z1 = torch.linspace(-grid_range, grid_range, grid_points)
        z2 = torch.linspace(-grid_range, grid_range, grid_points)
        z1_grid, z2_grid = torch.meshgrid(z1, z2, indexing='ij')
        z_grid = torch.stack([z1_grid.flatten(), z2_grid.flatten()], dim=1)

        # 生成图片
        generated_imgs = model.decoder(z_grid.to(device)).view(-1, 1, 28, 28)

        # 绘制网格图像
        fig, axes = plt.subplots(grid_points, grid_points, figsize=(12, 12))
        for i in range(grid_points):
            for j in range(grid_points):
                idx = i * grid_points + j
                axes[i, j].imshow(generated_imgs[idx].cpu().squeeze().numpy(), cmap='gray')
                axes[i, j].axis('off')

        plt.suptitle(f'Latent Space Grid (range: [-{grid_range}, {grid_range}])',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"潜在空间网格图已保存到: {save_path}")


def generate_interpolation(model, num_steps=10, save_path='vae_interpolation.png'):
    """在潜在空间两点之间插值生成"""
    with torch.no_grad():
        # 随机选择两个潜在向量
        z_start = torch.randn(1, model.latent_dim).to(device)
        z_end = torch.randn(1, model.latent_dim).to(device)

        # 线性插值
        alphas = torch.linspace(0, 1, num_steps)
        interpolated_z = torch.zeros(num_steps, model.latent_dim).to(device)
        for i, alpha in enumerate(alphas):
            interpolated_z[i] = (1 - alpha) * z_start + alpha * z_end

        # 生成图片
        generated_imgs = model.decoder(interpolated_z).view(-1, 1, 28, 28)

        # 绘制
        plt.figure(figsize=(15, 3))
        for i in range(num_steps):
            plt.subplot(1, num_steps, i + 1)
            plt.imshow(generated_imgs[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Start', fontsize=10)
            elif i == num_steps - 1:
                plt.title('End', fontsize=10)

        plt.suptitle('Latent Space Interpolation', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"插值图片已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='VAE 图片生成')
    parser.add_argument('--model', type=str, default='vae_model.pth',
                        help='模型权重路径 (默认: vae_model.pth)')
    parser.add_argument('--latent_dim', type=int, default=20,
                        help='潜在空间维度 (默认: 20)')
    parser.add_argument('--num_images', type=int, default=20,
                        help='随机生成的图片数量 (默认: 20)')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'grid', 'interpolation', 'all'],
                        help='生成模式: random(随机采样), grid(2D网格), interpolation(插值), all(全部)')
    parser.add_argument('--grid_range', type=float, default=3,
                        help='网格范围 (默认: ±3)')
    parser.add_argument('--grid_points', type=int, default=15,
                        help='网格点数 (默认: 15)')

    args = parser.parse_args()

    print("="*60)
    print("VAE 图片生成脚本")
    print("="*60)
    print(f"设备: {device}")
    print(f"模型路径: {args.model}")
    print(f"潜在空间维度: {args.latent_dim}")
    print("="*60)

    # 加载模型
    model = load_model(args.model, args.latent_dim)

    # 根据模式生成图片
    if args.mode in ['random', 'all']:
        print("\n[1/3] 随机采样生成...")
        generate_random_images(model, args.num_images)

    if args.mode in ['grid', 'all']:
        print("\n[2/3] 生成潜在空间网格...")
        generate_latent_grid(model, args.grid_range, args.grid_points)

    if args.mode in ['interpolation', 'all']:
        print("\n[3/3] 生成插值序列...")
        generate_interpolation(model)

    print("\n" + "="*60)
    print("生成完成!")
    print("="*60)


if __name__ == '__main__':
    main()
