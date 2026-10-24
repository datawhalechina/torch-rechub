"""HSTU Model Example on MovieLens Dataset."""

import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from torch_rechub.basic.metric import topk_metrics
from torch_rechub.models.generative.hstu import HSTUModel
from torch_rechub.trainers.seq_trainer import SeqTrainer
from torch_rechub.utils.data import SequenceDataGenerator


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
    item_to_idx = vocab['item_to_idx'] if 'item_to_idx' in vocab else vocab
    vocab_size = max(item_to_idx.values()) + 1

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


def main(
    dataset_path=None,
    model_name='hstu',
    epoch=101,
    learning_rate=1e-3,
    batch_size=128,
    weight_decay=0.0,
    device='cuda',
    save_dir='./',
    seed=2022,
    max_seq_len=200,
    d_model=50,
    n_heads=1,
    n_layers=2,
    dqk=50,
    dv=50,
    dropout=0.2,
    score_norm='l2',
    temperature=0.05,
    use_time_embedding=False,
    time_bucket_fn='log',
    time_bucket_divisor=0.301,
    time_bucket_unit='seconds',
    use_output_bias=False,
    scale_input_embedding=True,
    loss_type='cross_entropy',
    num_negatives=128,
    accumulation_steps=1,
    amp=False,
):
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

    print(f"\nDevice: {device}")
    print(f"Hidden dim: {d_model}")
    print(f"Num heads: {n_heads}")
    print(f"Num layers: {n_layers}")
    print(f"dqk/dv: {dqk}/{dv}")
    print(f"Dropout: {dropout}")
    print(f"Score norm: {score_norm}, temperature: {temperature}")
    print(f"Use input time embedding: {use_time_embedding}")
    print(f"Max sequence length: {max_seq_len}")

    # Load data
    print("\nLoading data...")
    result = get_movielens_data(dataset_path)
    if result is None:
        print("数据加载失败，退出...")
        return

    train_data, val_data, test_data, vocab_size = result

    # MovieLens preprocessing left-pads histories; retain the most recent events.
    for data in [train_data, val_data, test_data]:
        for key in ['seq_tokens', 'seq_positions', 'seq_time_diffs']:
            data[key] = np.asarray(data[key])[:, -max_seq_len:]

    # Build data loaders (with time-aware features)
    print("\nBuilding data loaders (with timestamp features)...")
    print("Using relative time/position bias")

    train_gen = SequenceDataGenerator(train_data['seq_tokens'], train_data['seq_positions'], train_data['targets'], train_data['seq_time_diffs'])
    val_gen = SequenceDataGenerator(val_data['seq_tokens'], val_data['seq_positions'], val_data['targets'], val_data['seq_time_diffs'])
    test_gen = SequenceDataGenerator(test_data['seq_tokens'], test_data['seq_positions'], test_data['targets'], test_data['seq_time_diffs'])

    train_dataloader = train_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
    val_dataloader = val_gen.generate_dataloader(batch_size=batch_size, num_workers=0, shuffle=False)[0]
    test_dataloader = test_gen.generate_dataloader(batch_size=batch_size, num_workers=0, shuffle=False)[0]

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
        dqk=dqk,
        dv=dv,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_time_embedding=use_time_embedding,
        time_bucket_fn=time_bucket_fn,
        time_bucket_divisor=time_bucket_divisor,
        time_bucket_unit=time_bucket_unit,
        score_norm=score_norm,
        temperature=temperature,
        use_output_bias=use_output_bias,
        scale_input_embedding=scale_input_embedding,
    )
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = SeqTrainer(
        model,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "betas": (0.9,
                      0.98),
        },
        n_epoch=epoch,
        earlystop_patience=max(epoch + 1,
                               10),
        device=device,
        model_path=save_dir,
        loss_type=loss_type,
        loss_params={'num_negatives': num_negatives} if loss_type == 'sampled_softmax' else None,
        accumulation_steps=accumulation_steps,
        amp=amp,
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
    topKs = sorted({min(k, vocab_size - 1) for k in [10, 50, 200]})
    predictions = trainer.predict(test_dataloader, top_k=max(topKs), filter_seen=True)
    y_true = {str(i): [int(target)] for i, target in enumerate(test_data['targets'])}
    y_pred = {str(i): items.tolist() for i, items in enumerate(predictions)}
    ranking_results = topk_metrics(y_true, y_pred, topKs=topKs)

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
    parser.add_argument('--epoch', type=int, default=101, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', default='./', help='Directory to save model')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=50, help='Hidden dimension')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of HSTU layers')
    parser.add_argument('--dqk', type=int, default=50, help='Query/key dimension per head')
    parser.add_argument('--dv', type=int, default=50, help='Value/U dimension per head')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--score_norm', default='l2', choices=['none', 'l2'], help='Output scoring normalization')
    parser.add_argument('--temperature', type=float, default=0.05, help='Output logit temperature')
    parser.add_argument('--use_time_embedding', action='store_true', help='Add input-side time bucket embeddings')
    parser.add_argument('--time_bucket_fn', default='log', choices=['sqrt', 'log'], help='Time bucketization function')
    parser.add_argument('--time_bucket_divisor', type=float, default=0.301, help='Time bucket divisor')
    parser.add_argument('--time_bucket_unit', default='seconds', choices=['seconds', 'minutes'], help='Unit for time bucketization')
    parser.add_argument('--use_output_bias', action='store_true', help='Use output bias in item logits')
    parser.add_argument('--no_scale_input_embedding', action='store_true', help='Disable sqrt(d_model) input embedding scale')
    parser.add_argument('--loss_type', default='cross_entropy', choices=['cross_entropy', 'sampled_softmax'])
    parser.add_argument('--num_negatives', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--amp', action='store_true')

    args = parser.parse_args()
    main(
        args.dataset_path,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.device,
        args.save_dir,
        args.seed,
        args.max_seq_len,
        args.d_model,
        args.n_heads,
        args.n_layers,
        args.dqk,
        args.dv,
        args.dropout,
        args.score_norm,
        args.temperature,
        args.use_time_embedding,
        args.time_bucket_fn,
        args.time_bucket_divisor,
        args.time_bucket_unit,
        args.use_output_bias,
        not args.no_scale_input_embedding,
        args.loss_type,
        args.num_negatives,
        args.accumulation_steps,
        args.amp,
    )
