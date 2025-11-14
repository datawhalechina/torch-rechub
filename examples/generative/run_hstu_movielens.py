"""HSTU Model Example on MovieLens Dataset."""

import torch
import numpy as np
import pickle
import os
from torch_rechub.models.generative.hstu import HSTUModel
from torch_rechub.utils.data import SequenceDataGenerator
from torch_rechub.trainers.seq_trainer import SeqTrainer


def load_real_data(data_dir="./data/ml-1m/processed/"):
    """加载真实的MovieLens-1M数据.

    Args:
        data_dir (str): 数据目录

    Returns:
        tuple: (train_data, val_data, test_data, vocab_size)
    """
    print("加载真实数据...")

    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请先运行: python examples/generative/data/ml-1m/preprocess_ml_hstu.py")
        return None

    # 加载词表
    vocab_file = os.path.join(data_dir, 'vocab.pkl')
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # 加载数据
    train_file = os.path.join(data_dir, 'train_data.pkl')
    val_file = os.path.join(data_dir, 'val_data.pkl')
    test_file = os.path.join(data_dir, 'test_data.pkl')

    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)

    print(f"数据加载完成: vocab_size={vocab_size}")
    return train_data, val_data, test_data, vocab_size


def generate_dummy_data(num_samples=1000, seq_len=256, vocab_size=10000):
    """生成虚拟数据用于测试.

    Args:
        num_samples (int): 样本数量
        seq_len (int): 序列长度
        vocab_size (int): 词表大小

    Returns:
        tuple: (seq_tokens, seq_positions, targets)
    """
    # 生成随机序列token
    seq_tokens = np.random.randint(2, vocab_size, size=(num_samples, seq_len))

    # 生成位置编码
    seq_positions = np.tile(np.arange(seq_len), (num_samples, 1))

    # 生成目标token
    targets = np.random.randint(2, vocab_size, size=(num_samples,))

    return seq_tokens, seq_positions, targets


def main(use_real_data=False):
    """主函数.

    Args:
        use_real_data (bool): 是否使用真实数据，默认False
    """
    print("=" * 80)
    print("HSTU Model Example on MovieLens")
    print("=" * 80)

    # 超参数
    d_model = 256
    n_heads = 8
    n_layers = 2
    batch_size = 32
    epochs = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n设备: {device}")
    print(f"模型维度: {d_model}")
    print(f"多头数: {n_heads}")
    print(f"层数: {n_layers}")

    # 加载数据
    print("\n加载数据...")
    if use_real_data:
        result = load_real_data()
        if result is None:
            print("使用虚拟数据代替...")
            use_real_data = False
        else:
            train_data, val_data, test_data, vocab_size = result

    if not use_real_data:
        print("生成虚拟数据...")
        vocab_size = 10000
        seq_tokens, seq_positions, targets = generate_dummy_data(
            num_samples=1000,
            seq_len=256,
            vocab_size=vocab_size
        )
        print(f"数据形状: seq_tokens={seq_tokens.shape}, targets={targets.shape}")

        # 创建数据加载器
        print("\n创建数据加载器...")
        data_gen = SequenceDataGenerator(seq_tokens, seq_positions, targets)
        train_loader, val_loader, test_loader = data_gen.generate_dataloader(
            batch_size=batch_size,
            num_workers=0,
            split_ratio=(0.7, 0.1, 0.2)
        )
    else:
        # 使用真实数据
        print("\n创建数据加载器（真实数据）...")
        train_gen = SequenceDataGenerator(
            train_data['seq_tokens'],
            train_data['seq_positions'],
            train_data['targets']
        )
        val_gen = SequenceDataGenerator(
            val_data['seq_tokens'],
            val_data['seq_positions'],
            val_data['targets']
        )
        test_gen = SequenceDataGenerator(
            test_data['seq_tokens'],
            test_data['seq_positions'],
            test_data['targets']
        )

        train_loader = train_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
        val_loader = val_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
        test_loader = test_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]

    print(f"词表大小: {vocab_size}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = HSTUModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=256,
        dropout=0.1
    )
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = SeqTrainer(model, optimizer, device=device)
    
    # 训练模型
    print("\n开始训练...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=2,
        save_path='hstu_model.pt'
    )
    
    # 评估模型
    print("\n评估模型...")
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 打印训练历史
    print("\n训练历史:")
    for epoch in range(len(history['train_loss'])):
        print(f"Epoch {epoch+1}: "
              f"train_loss={history['train_loss'][epoch]:.4f}, "
              f"val_loss={history['val_loss'][epoch]:.4f}, "
              f"val_accuracy={history['val_accuracy'][epoch]:.4f}")
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HSTU Model Example on MovieLens')
    parser.add_argument('--use-real-data', action='store_true',
                        help='使用真实MovieLens-1M数据')
    args = parser.parse_args()

    main(use_real_data=args.use_real_data)

