"""HSTU Model Example on MovieLens Dataset."""

import os
import pickle
import sys

import numpy as np
import torch
import tqdm

from torch_rechub.basic.metric import topk_metrics
from torch_rechub.models.generative.hstu import HSTUModel
from torch_rechub.trainers.seq_trainer import SeqTrainer
from torch_rechub.utils.data import SequenceDataGenerator

sys.path.append("../..")


def get_movielens_data(data_dir="./data/ml-1m/processed/"):
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


def evaluate_ranking(model, data_loader, device, topKs=[10, 50, 200]):
    """评估推荐排序指标.

    Args:
        model: 训练好的模型
        data_loader: 测试数据加载器
        device: 设备
        topKs: 评估的K值列表

    Returns:
        dict: 包含各种推荐指标的字典
    """
    model.eval()
    y_true = {}
    y_pred = {}

    user_idx = 0
    with torch.no_grad():
        for seq_tokens, seq_positions, seq_time_diffs, targets in tqdm.tqdm(data_loader, desc="evaluating ranking", smoothing=0, mininterval=1.0):
            # 移动到设备
            seq_tokens = seq_tokens.to(device)
            seq_time_diffs = seq_time_diffs.to(device)
            targets = targets.cpu().numpy()

            # 前向传播
            logits = model(seq_tokens, seq_time_diffs)  # (B, L, V)

            # 对于next-item prediction任务，只使用最后一个位置的预测
            last_logits = logits[:, -1, :]  # (B, V)

            # 获取每个样本的top-K推荐
            batch_size = last_logits.shape[0]
            max_k = max(topKs)

            # 获取top-K的物品索引
            _, top_items = torch.topk(last_logits, k=max_k, dim=-1)  # (B, max_k)
            top_items = top_items.cpu().numpy()

            # 为每个样本构建y_true和y_pred
            for i in range(batch_size):
                user_id = str(user_idx)
                y_true[user_id] = [int(targets[i])]  # 真实的下一个物品
                y_pred[user_id] = top_items[i].tolist()  # 推荐的top-K物品
                user_idx += 1

    # 计算推荐指标
    results = topk_metrics(y_true, y_pred, topKs=topKs)
    return results


def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, max_seq_len):
    """主函数.

    Args:
        dataset_path (str): 数据集路径
        model_name (str): 模型名称
        epoch (int): 训练轮数
        learning_rate (float): 学习率
        batch_size (int): 批次大小
        weight_decay (float): 权重衰减
        device (str): 设备
        save_dir (str): 模型保存目录
        seed (int): 随机种子
        max_seq_len (int): 最大序列长度
    """
    torch.manual_seed(seed)

    print("=" * 80)
    print("HSTU Model Example on MovieLens")
    print("=" * 80)

    # 超参数
    d_model = 256
    n_heads = 8
    n_layers = 2

    print(f"\n设备: {device}")
    print(f"模型维度: {d_model}")
    print(f"多头数: {n_heads}")
    print(f"层数: {n_layers}")
    print(f"序列长度: {max_seq_len}")

    # 加载数据
    print("\n加载数据...")
    result = get_movielens_data(dataset_path)
    if result is None:
        print("数据加载失败，退出...")
        return

    train_data, val_data, test_data, vocab_size = result

    # 使用真实数据（支持时间差）
    print("\n创建数据加载器（真实数据）...")
    print("✅ 使用时间感知的位置编码")

    train_gen = SequenceDataGenerator(
        train_data['seq_tokens'],
        train_data['seq_positions'],
        train_data['targets'],
        train_data['seq_time_diffs']
    )
    val_gen = SequenceDataGenerator(
        val_data['seq_tokens'],
        val_data['seq_positions'],
        val_data['targets'],
        val_data['seq_time_diffs']
    )
    test_gen = SequenceDataGenerator(
        test_data['seq_tokens'],
        test_data['seq_positions'],
        test_data['targets'],
        test_data['seq_time_diffs']
    )

    train_dataloader = train_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
    val_dataloader = val_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
    test_dataloader = test_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]

    print(f"词表大小: {vocab_size}")
    print(f"训练集大小: {len(train_dataloader.dataset)}")
    print(f"验证集大小: {len(val_dataloader.dataset)}")
    print(f"测试集大小: {len(test_dataloader.dataset)}")

    # 创建模型
    print("\n创建模型...")
    model = HSTUModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器
    print("\n创建训练器...")
    trainer = SeqTrainer(
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
        n_epoch=epoch,
        earlystop_patience=10,
        device=device,
        model_path=save_dir
    )

    # 训练模型
    print("\n开始训练...")
    trainer.fit(train_dataloader, val_dataloader)

    # 评估模型
    print("\n评估模型...")
    test_loss, test_accuracy = trainer.evaluate(test_dataloader)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 评估推荐指标
    print("\n计算推荐指标...")
    ranking_results = evaluate_ranking(model, test_dataloader, device, topKs=[10, 50, 200])

    print("\n测试集推荐指标:")
    print("=" * 50)
    # 提取并打印HR和NDCG指标
    for metric_name in ['Hit', 'NDCG']:
        for result_str in ranking_results[metric_name]:
            print(result_str)
    print("=" * 50)

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/ml-1m/processed/")
    parser.add_argument('--model_name', default='hstu')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')  # cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--max_seq_len', type=int, default=200)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size,
         args.weight_decay, args.device, args.save_dir, args.seed, args.max_seq_len)