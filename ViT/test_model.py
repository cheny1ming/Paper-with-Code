"""
快速测试脚本 - 验证ViT模型是否正常工作
"""

import torch
from ViT import VisionTransformer, ViTConfig


def test_basic_model():
    """测试基础模型"""
    print("="*60)
    print("Testing Basic ViT Model")
    print("="*60)

    try:
        # 创建模型
        config = ViTConfig.vit_base()
        config.num_classes = 1000

        model = VisionTransformer(
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads
        )

        # 测试前向传播
        image = torch.randn(2, 3, 224, 224)
        output = model(image)
        features = model.forward_features(image)

        print(f"✓ Input shape: {image.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Features shape: {features.shape}")
        print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("✓ Basic model test PASSED")
        return True

    except Exception as e:
        print(f"✗ Basic model test FAILED: {str(e)}")
        return False


def test_model_sizes():
    """测试不同尺寸的模型"""
    print("\n" + "="*60)
    print("Testing Different Model Sizes")
    print("="*60)

    sizes = {
        'ViT-Tiny': ViTConfig.vit_tiny(),
        'ViT-Small': ViTConfig.vit_small(),
        'ViT-Base': ViTConfig.vit_base(),
        'ViT-Large': ViTConfig.vit_large(),
        'ViT-Huge': ViTConfig.vit_huge()
    }

    for name, config in sizes.items():
        try:
            config.num_classes = 1000

            model = VisionTransformer(
                img_size=config.img_size,
                patch_size=config.patch_size,
                num_classes=config.num_classes,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads
            )

            params = sum(p.numel() for p in model.parameters())

            # 测试前向传播
            image = torch.randn(1, 3, config.img_size, config.img_size)
            output = model(image)

            print(f"✓ {name}: {params:,} parameters, output shape: {output.shape}")

        except Exception as e:
            print(f"✗ {name} test FAILED: {str(e)}")
            return False

    print("✓ Model size test PASSED")
    return True


def test_different_input_sizes():
    """测试不同输入尺寸"""
    print("\n" + "="*60)
    print("Testing Different Input Sizes")
    print("="*60)

    input_sizes = [224, 256, 384]

    for size in input_sizes:
        try:
            model = VisionTransformer(
                img_size=size,
                patch_size=16,
                num_classes=1000
            )

            # 测试前向传播
            image = torch.randn(2, 3, size, size)
            output = model(image)

            print(f"✓ Input size {size}x{size}: output shape {output.shape}")

        except Exception as e:
            print(f"✗ Input size {size}x{size} test FAILED: {str(e)}")
            return False

    print("✓ Input size test PASSED")
    return True


def test_feature_extraction():
    """测试特征提取"""
    print("\n" + "="*60)
    print("Testing Feature Extraction")
    print("="*60)

    try:
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=1000
        )

        # 测试特征提取
        images = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            features = model.forward_features(images)

        print(f"✓ Input shape: {images.shape}")
        print(f"✓ Features shape: {features.shape}")
        print(f"✓ Feature dimension: {features.shape[-1]}")
        print("✓ Feature extraction test PASSED")
        return True

    except Exception as e:
        print(f"✗ Feature extraction test FAILED: {str(e)}")
        return False


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "="*60)
    print("Testing Batch Processing")
    print("="*60)

    try:
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=10
        )

        model.eval()

        # 测试不同的batch size
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            images = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                outputs = model(images)

            print(f"✓ Batch size {batch_size}: output shape {outputs.shape}")

        print("✓ Batch processing test PASSED")
        return True

    except Exception as e:
        print(f"✗ Batch processing test FAILED: {str(e)}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Vision Transformer Model Test Suite")
    print("="*60)

    # 检查PyTorch是否可用
    if not torch.cuda.is_available():
        print("⚠ Warning: CUDA not available, using CPU")
    else:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    # 运行测试
    results = []
    results.append(("Basic Model", test_basic_model()))
    results.append(("Model Sizes", test_model_sizes()))
    results.append(("Input Sizes", test_different_input_sizes()))
    results.append(("Feature Extraction", test_feature_extraction()))
    results.append(("Batch Processing", test_batch_processing()))

    # 总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Model is ready to use.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
