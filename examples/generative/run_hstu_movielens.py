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

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_movielens_data(data_dir=None):
    """Load preprocessed MovieLens-1M data from disk.

    Args:
        data_dir (str): Directory containing preprocessed files.
                       If None, uses default path relative to script location.

    Returns:
        tuple: ``(train_data, val_data, test_data, vocab_size)``.
    """
    print("加载真实数据...")

    # 如果没有指定数据目录，使用脚本所在目录的相对路径
    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data/ml-1m/processed/")
    elif not os.path.isabs(data_dir):
        # 如果是相对路径，相对于脚本所在目录
        data_dir = os.path.join(SCRIPT_DIR, data_dir)

    # 转换为绝对路径
    data_dir = os.path.abspath(data_dir)

    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("\n请先运行数据预处理脚本:")
        print(f"  cd {os.path.join(SCRIPT_DIR, 'data/ml-1m')}")
        print("  python preprocess_ml_hstu.py")
        return None

    # 加载词表
    vocab_file = os.path.join(data_dir, 'vocab.pkl')
    if not os.path.exists(vocab_file):
        print(f"❌ 词表文件不存在: {vocab_file}")
        print("请先运行数据预处理脚本")
        return None

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # 加载数据
    train_file = os.path.join(data_dir, 'train_data.pkl')
    val_file = os.path.join(data_dir, 'val_data.pkl')
    test_file = os.path.join(data_dir, 'test_data.pkl')

    required_files = [train_file, val_file, test_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            print("请先运行数据预处理脚本")
            return None

    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)

    print(f"✅ 数据加载完成: vocab_size={vocab_size}")
    print(f"   数据目录: {data_dir}")
    return train_data, val_data, test_data, vocab_size


def evaluate_ranking(model, data_loader, device, topKs=[10, 50, 200]):
    """Evaluate top-K ranking metrics on the test set.

    Args:
        model: Trained recommendation model.
        data_loader: DataLoader providing test sequences.
        device: Target device for inference.
        topKs: List of K values to evaluate (e.g. [10, 50, 200]).

    Returns:
        dict: A mapping from metric name to formatted result strings.
    """
    model.eval()
    y_true = {}
    y_pred = {}

    user_idx = 0
    with torch.no_grad():
        for seq_tokens, seq_positions, seq_time_diffs, targets in tqdm.tqdm(data_loader, desc="evaluating ranking", smoothing=0, mininterval=1.0):
            # Move tensors to device
            seq_tokens = seq_tokens.to(device)
            seq_time_diffs = seq_time_diffs.to(device)
            targets = targets.cpu().numpy()

            # Forward pass
            logits = model(seq_tokens, seq_time_diffs)  # (B, L, V)

            # For next-item prediction, only use the last position
            last_logits = logits[:, -1, :]  # (B, V)

            # Get top-K recommendations for each sample
            batch_size = last_logits.shape[0]
            max_k = max(topKs)

            # Indices of top-K items
            _, top_items = torch.topk(last_logits, k=max_k, dim=-1)  # (B, max_k)
            top_items = top_items.cpu().numpy()

            # Build y_true and y_pred for each sample
            for i in range(batch_size):
                user_id = str(user_idx)
                y_true[user_id] = [int(targets[i])]  # ground-truth next item
                y_pred[user_id] = top_items[i].tolist()  # predicted top-K items
                user_idx += 1

    # Compute ranking metrics
    results = topk_metrics(y_true, y_pred, topKs=topKs)
    return results


def main(dataset_path=None, model_name='hstu', epoch=5, learning_rate=1e-3, batch_size=512, weight_decay=1e-5, device='cuda', save_dir='./', seed=2022, max_seq_len=200):
    """Main training and evaluation entry point.

    Args:
        dataset_path (str): Path to preprocessed MovieLens data.
        model_name (str): Model name (kept for CLI consistency).
        epoch (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size.
        weight_decay (float): Weight decay (L2 regularization).
        device (str): Device spec, e.g. ``"cuda"`` or ``"cpu"``.
        save_dir (str): Directory to save model checkpoints.
        seed (int): Random seed for reproducibility.
        max_seq_len (int): Maximum sequence length.
    """
    torch.manual_seed(seed)

    print("=" * 80)
    print("HSTU Model Example on MovieLens")
    print("=" * 80)

    # Model hyper-parameters
    d_model = 256
    n_heads = 8
    n_layers = 2

    print(f"\nDevice: {device}")
    print(f"Hidden dim: {d_model}")
    print(f"Num heads: {n_heads}")
    print(f"Num layers: {n_layers}")
    print(f"Max sequence length: {max_seq_len}")

    # Load data
    print("\nLoading data...")
    result = get_movielens_data(dataset_path)
    if result is None:
        print("数据加载失败，退出...")
        return

    train_data, val_data, test_data, vocab_size = result

    # Build data loaders (with time-aware features)
    print("\nBuilding data loaders (with time-aware features)...")
    print("✅ Using time-aware positional encoding")

    train_gen = SequenceDataGenerator(train_data['seq_tokens'], train_data['seq_positions'], train_data['targets'], train_data['seq_time_diffs'])
    val_gen = SequenceDataGenerator(val_data['seq_tokens'], val_data['seq_positions'], val_data['targets'], val_data['seq_time_diffs'])
    test_gen = SequenceDataGenerator(test_data['seq_tokens'], test_data['seq_positions'], test_data['targets'], test_data['seq_time_diffs'])

    train_dataloader = train_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
    val_dataloader = val_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
    test_dataloader = test_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]

    print(f"Vocab size: {vocab_size}")
    print(f"Train size: {len(train_dataloader.dataset)}")
    print(f"Val size: {len(val_dataloader.dataset)}")
    print(f"Test size: {len(test_dataloader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = HSTUModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = SeqTrainer(
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr": learning_rate,
            "weight_decay": weight_decay
        },
        n_epoch=epoch,
        earlystop_patience=10,
        device=device,
        model_path=save_dir,
    )

    # Train model
    print("\nStart training...")
    trainer.fit(train_dataloader, val_dataloader)

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = trainer.evaluate(test_dataloader)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Evaluate ranking metrics
    print("\nComputing ranking metrics...")
    ranking_results = evaluate_ranking(model, test_dataloader, device, topKs=[10, 50, 200])

    print("\nRanking metrics on the test set:")
    print("=" * 50)
    # Print HR and NDCG metrics
    for metric_name in ["Hit", "NDCG"]:
        for result_str in ranking_results[metric_name]:
            print(result_str)
    print("=" * 50)

    print("\n" + "=" * 80)
    print("Training finished!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HSTU Model Training on MovieLens-1M Dataset')
    parser.add_argument('--dataset_path', default=None, help='Path to preprocessed data directory (default: auto-detect relative to script)')
    parser.add_argument('--model_name', default='hstu', help='Model name')
    parser.add_argument('--epoch', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', default='./', help='Directory to save model')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Maximum sequence length')

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed, args.max_seq_len)
