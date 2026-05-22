"""HSTU Model Example on Amazon Books Dataset.

Notes on memory:
    Amazon Books has ~686k items, so the per-batch logits tensor
    ``(B, L, V)`` is large. Defaults here (``d_model=64``, ``batch_size=64``,
    ``max_seq_len=50``) are tuned to fit on a 24 GB GPU. Scale them up on
    larger cards (e.g. ``--d_model 256 --batch_size 128``).
"""

import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import tqdm

from torch_rechub.basic.metric import topk_metrics
from torch_rechub.models.generative.hstu import HSTUModel
from torch_rechub.trainers.seq_trainer import SeqTrainer
from torch_rechub.utils.data import SequenceDataGenerator, pad_sequences
from torch_rechub.utils.hstu_utils import VocabMask


def get_amazon_books_data(data_dir=None, max_seq_len=50):
    """Load preprocessed Amazon Books data and pad to fixed length.

    Unlike the MovieLens HSTU pipeline (which writes pre-padded numpy
    arrays), :mod:`preprocess_amazon_books` stores variable-length Python
    lists. We pre-pad / pre-truncate to ``max_seq_len`` here so that the
    standard ``SequenceDataGenerator`` can stack batches.

    Args:
        data_dir (str): Directory containing preprocessed files.
                       If None, uses default path relative to script location.
        max_seq_len (int): Target sequence length after padding/truncation.

    Returns:
        tuple: ``(train_data, val_data, test_data, vocab_size)``.
    """
    print("加载真实数据...")

    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data/amazon-books/processed/")
    elif not os.path.isabs(data_dir):
        data_dir = os.path.join(SCRIPT_DIR, data_dir)

    data_dir = os.path.abspath(data_dir)

    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("\n请先运行数据预处理脚本:")
        print(f"  cd {os.path.join(SCRIPT_DIR, 'data/amazon-books')}")
        print("  python preprocess_amazon_books.py --data_source bytedance")
        return None

    vocab_file = os.path.join(data_dir, 'vocab.pkl')
    if not os.path.exists(vocab_file):
        print(f"❌ 词表文件不存在: {vocab_file}")
        print("请先运行数据预处理脚本")
        return None

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    item_to_idx = vocab['item_to_idx'] if 'item_to_idx' in vocab else vocab
    vocab_size = max(item_to_idx.values()) + 1

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

    # Pad variable-length sequences. ``pre`` keeps the most recent items.
    for data in (train_data, val_data, test_data):
        data['seq_tokens'] = pad_sequences(data['seq_tokens'], maxlen=max_seq_len, padding='pre', truncating='pre', value=0)
        data['seq_positions'] = pad_sequences(data['seq_positions'], maxlen=max_seq_len, padding='pre', truncating='pre', value=0)
        data['seq_time_diffs'] = pad_sequences(data['seq_time_diffs'], maxlen=max_seq_len, padding='pre', truncating='pre', value=0)
        data['targets'] = np.array(data['targets'])

    print(f"✅ 数据加载完成: vocab_size={vocab_size}")
    print(f"   数据目录: {data_dir}")
    return train_data, val_data, test_data, vocab_size


def evaluate_ranking(model, data_loader, device, topKs=[10, 50, 200], invalid_items=(0,), filter_seen=True):
    """Evaluate top-K ranking metrics on the test set.

    PAD (token id 0), any ids in ``invalid_items``, and optionally each user's
    historical items are masked out before top-K. Filtering history matches the
    Meta public evaluation protocol for HSTU/SASRec.
    """
    model = model.to(device)
    model.eval()
    vocab_mask = VocabMask(model.vocab_size, invalid_items=list(invalid_items)).to(device)

    y_true = {}
    y_pred = {}

    user_idx = 0
    with torch.no_grad():
        for seq_tokens, _seq_positions, seq_time_diffs, targets in tqdm.tqdm(data_loader, desc="evaluating ranking", smoothing=0, mininterval=1.0):
            seq_tokens = seq_tokens.to(device)
            seq_time_diffs = seq_time_diffs.to(device)
            targets = targets.cpu().numpy()

            logits = model(seq_tokens, seq_time_diffs)  # (B, L, V)
            seen_ids = seq_tokens if filter_seen else None
            last_logits = vocab_mask.apply_mask(logits[:, -1, :], invalid_ids=seen_ids)  # (B, V)

            max_k = max(topKs)
            _, top_items = torch.topk(last_logits, k=max_k, dim=-1)
            top_items = top_items.cpu().numpy()

            for i in range(top_items.shape[0]):
                user_id = str(user_idx)
                y_true[user_id] = [int(targets[i])]
                y_pred[user_id] = top_items[i].tolist()
                user_idx += 1

    return topk_metrics(y_true, y_pred, topKs=topKs)


def main(dataset_path=None, model_name='hstu', epoch=3, learning_rate=1e-3, batch_size=64, weight_decay=1e-5, device='cuda', save_dir='./', seed=2022, max_seq_len=50, d_model=64, n_heads=4, n_layers=2):
    """Main training and evaluation entry point.

    Args:
        dataset_path (str): Path to preprocessed Amazon Books data.
        model_name (str): Model name (kept for CLI consistency).
        epoch (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size.
        weight_decay (float): Weight decay (L2 regularization).
        device (str): Device spec, e.g. ``"cuda"`` or ``"cpu"``.
        save_dir (str): Directory to save model checkpoints.
        seed (int): Random seed for reproducibility.
        max_seq_len (int): Maximum sequence length.
        d_model (int): Hidden dimension.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of HSTU layers.
    """
    torch.manual_seed(seed)

    print("=" * 80)
    print("HSTU Model Example on Amazon Books")
    print("=" * 80)

    print(f"\nDevice: {device}")
    print(f"Hidden dim: {d_model}")
    print(f"Num heads: {n_heads}")
    print(f"Num layers: {n_layers}")
    print(f"Max sequence length: {max_seq_len}")

    print("\nLoading data...")
    result = get_amazon_books_data(dataset_path, max_seq_len=max_seq_len)
    if result is None:
        print("数据加载失败，退出...")
        return

    train_data, val_data, test_data, vocab_size = result

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

    print("\nStart training...")
    trainer.fit(train_dataloader, val_dataloader)

    print("\nEvaluating model...")
    test_loss, test_accuracy = trainer.evaluate(test_dataloader)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nComputing ranking metrics...")
    ranking_results = evaluate_ranking(model, test_dataloader, device, topKs=[10, 50, 200])

    print("\nRanking metrics on the test set:")
    print("=" * 50)
    for metric_name in ["Hit", "NDCG"]:
        for result_str in ranking_results[metric_name]:
            print(result_str)
    print("=" * 50)

    print("\n" + "=" * 80)
    print("Training finished!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HSTU Model Training on Amazon Books Dataset')
    parser.add_argument('--dataset_path', default=None, help='Path to preprocessed data directory (default: auto-detect relative to script)')
    parser.add_argument('--model_name', default='hstu', help='Model name')
    parser.add_argument('--epoch', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', default='./', help='Directory to save model')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--max_seq_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of HSTU layers')

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed, args.max_seq_len, args.d_model, args.n_heads, args.n_layers)
