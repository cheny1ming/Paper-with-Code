# CLIP (Contrastive Language-Image Pre-training) Implementation

完整的端到端CLIP算法实现，基于论文 "Learning Transferable Visual Models From Natural Language Supervision"。
这是我的Blog: [CLIP](https://cheny1ming.github.io/Blogs/post.html?id=clip)

## 实现内容:
- ✅ Image Encoder (基于 ViT)
- ✅ Text Encoder (基于 Transformer)
- ✅ Contrastive Loss 对比学习
- ✅ 多模态特征对齐
- ✅ Zero-shot 图像分类
- ✅ 图文检索

## 项目结构

```
CLIP/
├── image_encoder.py    # Vision Transformer 图像编码器
├── text_encoder.py     # Transformer 文本编码器
├── CLIP.py             # 端到端CLIP模型 + 对比损失
├── train.py            # 训练示例脚本
└── README.md           # 本文件
```

## 核心组件

### 1. Image Encoder ([image_encoder.py](image_encoder.py))
- **Vision Transformer (ViT)**: 将图像分割成patches，通过Transformer编码
- **Patch Embedding**: 使用卷积层将图像patches转换为嵌入
- **Multi-Head Attention**: 自注意力机制
- **Position Embedding**: 位置编码
- **L2 Normalization**: 输出特征归一化，便于余弦相似度计算

### 2. Text Encoder ([text_encoder.py](text_encoder.py))
- **Transformer**: 标准Transformer编码器
- **Token Embedding**: 词嵌入
- **Causal Attention**: 因果注意力掩码
- **EOS Token Extraction**: 使用EOS token作为句子级表示
- **L2 Normalization**: 输出特征归一化

### 3. CLIP Model ([CLIP.py](CLIP.py))

#### CLIPLoss
对称的对比学习损失：
```
Loss = (CE(I->T) + CE(T->I)) / 2
```
其中CE是交叉熵损失，I是图像，T是文本。

#### 主要功能
- **forward()**: 计算对比损失
- **encode_image()**: 编码图像
- **encode_text()**: 编码文本
- **compute_similarity()**: 计算图像-文本相似度矩阵
- **retrieve_text()**: 检索最匹配的文本
- **retrieve_images()**: 检索最匹配的图像
- **zero_shot_classification()**: 零样本图像分类

### 4. Training Script ([train.py](train.py))
完整的训练循环示例：
- AdamW优化器 + Cosine学习率调度
- 批量训练和验证
- 模型检查点保存
- 指标追踪（损失、准确率）

## 使用方法

### 基础用法

```python
from CLIP import CLIP
import torch

# 创建模型
model = CLIP(
    embed_dim=512,
    image_embed_dim=768,
    text_embed_dim=512,
    vision_depth=12,
    text_depth=12
)

# 前向传播
images = torch.randn(4, 3, 224, 224)
texts = torch.randint(0, 49408, (4, 77))
loss, metrics = model(images, texts)
```

### 零样本分类

```python
# 图像分类
class_prompts = ["a dog", "a cat", "a bird"]  # 编码后
probs = model.zero_shot_classification(images, class_prompts)
predicted_classes = probs.argmax(dim=-1)
```

### 图文检索

```python
# 检索最匹配的文本
scores, indices = model.retrieve_text(
    query_images,
    candidate_texts,
    top_k=5
)
```

### 训练

```bash
python train.py
```

## 模型架构

```
输入图像 (3×224×224)
    ↓
Vision Transformer
    ↓
[CLS] token (768维)
    ↓
Projection + L2 Norm
    ↓
图像特征 (512维)

输入文本 (77 tokens)
    ↓
Transformer
    ↓
EOS token (512维)
    ↓
Projection + L2 Norm
    ↓
文本特征 (512维)

图像特征 @ 文本特征 → 相似度矩阵 → 对比损失
```

## 关键特性

1. **L2归一化**: 图像和文本特征都经过L2归一化，使得相似度计算等同于余弦相似度
2. **可学习温度**: 使用 `logit_scale` 参数控制logits的锐度
3. **对称损失**: 同时计算图像→文本和文本→图像的交叉熵损失
4. **多任务支持**: 支持检索、分类、相似度计算等多种任务

## 依赖

```bash
pip install torch torchvision numpy pillow
```

## 扩展建议

1. **使用真实Tokenizer**: 集成SentencePiece或BERT tokenizer
2. **数据增强**: 添加RandAugment等图像增强
3. **分布式训练**: 使用DDP进行多GPU训练
4. **混合精度**: 使用AMP加速训练
5. **更大模型**: 增加depth和embed_dim以获得更好性能

## 参考

- [CLIP论文](https://arxiv.org/abs/2103.00020)
- [Vision Transformer论文](https://arxiv.org/abs/2010.11929)