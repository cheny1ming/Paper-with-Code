# Paper with Code

从论文到代码的实现仓库。精选深度学习经典论文，提供从零开始的完整代码实现。

## 📚 项目简介

本项目旨在深入理解深度学习领域的经典论文，通过从零开始实现核心模型，帮助学习者和开发者更好地掌握前沿技术。每个模型都包含：

- 📖 完整的代码实现
- 💡 详细的代码注释
- 🎯 端到端的训练流程
- 📝 配套的博客解析

## ✅ 已完成章节

### 1. Vision Transformer (ViT)

**论文**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

**博客**: [ViT 模型详解](https://cheny1ming.github.io/Blogs/post.html?id=vit)

**目录**: [ViT/](ViT/)

## 🚧 更新计划

### 近期计划

#### 1. CLIP (Contrastive Language-Image Pre-training)
**论文**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

**预计实现内容**:
- ⬜ Image Encoder (基于 ViT/CNN)
- ⬜ Text Encoder (基于 Transformer)
- ⬜ Contrastive Loss 对比学习
- ⬜ 多模态特征对齐
- ⬜ Zero-shot 图像分类
- ⬜ 图文检索

**状态**: 📋 准备中

#### 2. Swin Transformer
**论文**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

**预计实现内容**:
- ⬜ Shifted Window Attention
- ⬜ Hierarchical Feature Extraction
- ⬜ Patch Merging
- ⬜ 多尺度特征提取
- ⬜ 相对位置编码
- ⬜ 多种模型配置 (Tiny/Base/Large)

**状态**: 📅 计划中

#### 3. Stable Diffusion
**论文**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

**预计实现内容**:
- ⬜ VAE (Variational Autoencoder)
- ⬜ U-Net with Attention
- ⬜ CLIP 文本编码器集成
- ⬜ 扩散过程 (Forward/Reverse)
- ⬜ Text-to-Image 生成
- ⬜ Classifier-Free Guidance

**状态**: 📅 计划中

#### 4. VideoMAE (视频理解)
**论文**: [VideoMAE: Masked Autoencoders for Video Prediction](https://arxiv.org/abs/2203.12602)

**预计实现内容**:
- ⬜ 3D Cube Embedding
- ⬜ Tube Masking 策略
- ⬜ 多尺度时空特征提取
- ⬜ 视频动作识别
- ⬜ 高效的视频预训练方法
- ⬜ 微调下游任务

**状态**: 📅 计划中

#### 5. Video Diffusion (视频生成)
**论文**: [Video Diffusion Models](https://arxiv.org/abs/2204.03458)

**预计实现内容**:
- ⬜ 3D U-Net 架构
- ⬜ 时序扩散过程
- ⬜ Video-VAE 潜空间编码
- ⬜ 条件视频生成
- ⬜ Text-to-Video 生成
- ⬜ 图像到视频生成

**状态**: 📅 计划中

#### 6. AnimateDiff (动画生成)
**论文**: [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models](https://arxiv.org/abs/2307.04725)

**预计实现内容**:
- ⬜ Motion Module 运动模块
- ⬜ 时序注意力机制
- ⬜ 轻量级视频生成插件
- ⬜ 与 Stable Diffusion 集成
- ⬜ 个性化动画生成
- ⬜ 多种生成风格

**状态**: 📅 计划中

### 长期规划

- [ ] GPT 系列 (GPT-2, GPT-3)
- [ ] RLHF (PPO, GRPO, DAPO, GSPO...)
- [ ] SAM (Segment Anything Model)
- [ ] MAE (Masked Autoencoders)
- [ ] Sora - 大规模视频生成模型
- [ ] 开源多模态大模型 Qwen, GLM, Kimi...

## 📖 学习路线

```
基础模型
├── Vision Transformer (ViT) ✅
│   └── 理解 Transformer 在视觉中的应用
│
├── CLIP (进行中)
│   └── 学习多模态对比学习
│
├── Swin Transformer (计划中)
│   └── 掌握层次化视觉 Transformer
│
└── Stable Diffusion (计划中)
    └── 掌握生成模型与扩散过程

视频理解与生成
├── VideoMAE - 视频自监督学习
├── Video Diffusion - 视频扩散模型
└── AnimateDiff - 动画生成模型
```


## 🎯 使用建议

### 学习路径建议

1. **初学者**: 从 ViT 开始，理解 Transformer 在视觉任务中的应用
2. **进阶学习**: 研究 CLIP，掌握多模态学习
3. **视觉进阶**: 学习 Swin Transformer，理解层次化设计
4. **生成模型**: 学习 Stable Diffusion，理解扩散过程
5. **视频理解**: 研究 VideoMAE，掌握视频自监督学习
6. **视频生成**: 学习 Video Diffusion 和 AnimateDiff，理解视频生成

### 代码使用建议

- 每个模型目录都有独立的 `requirements.txt`
- 建议为每个模型创建独立的虚拟环境
- 先运行 `test_model.py` 验证环境配置
- 参考 `QUICKSTART.md` 快速上手

## 📊 进度追踪

| 模型 | 论文阅读 | 代码实现 | 测试验证 | 文档完善 | 博客发布 |
|------|----------|----------|----------|----------|----------|
| ViT  | ✅ | ✅ | ✅ | ✅ | ✅ |
| CLIP | ✅ | 🚧 | ⬜ | ⬜ | ⬜ |
| Swin Transformer | ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| Stable Diffusion | ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| VideoMAE | ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| Video Diffusion | ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| AnimateDiff | ✅ | ⬜ | ⬜ | ⬜ | ⬜ |

## 🔗 相关资源

- [我的博客](https://cheny1ming.github.io/Blogs/)
- [Papers with Code](https://paperswithcode.com/)
- [Hugging Face Models](https://huggingface.co/models)

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如果你有想要实现的论文或改进建议，欢迎提出。

## 📮 联系方式

- 博客: [cheny1ming.github.io](https://cheny1ming.github.io/Blogs/)
- GitHub: [@cheny1ming](https://github.com/cheny1ming)

---

⭐ 如果这个项目对你有帮助，欢迎 Star 支持一下！

最后更新: 2026-03-12