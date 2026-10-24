"""HLLM Model Example on MovieLens Dataset.

Architecture Overview:
- Item Embeddings: Pre-computed using LLM (offline)
- User encoder: small, randomly initialized Transformer blocks (trainable)
- Loss: NCE Loss on model-scaled cos-sim logits

This is a lightweight implementation that uses pre-computed item embeddings
instead of the full end-to-end training with Item LLM.
"""

import os
import pickle
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from torch_rechub.basic.metric import topk_metrics
from torch_rechub.models.generative.hllm import HLLMModel
from torch_rechub.trainers.seq_trainer import SeqTrainer
from torch_rechub.utils.data import SequenceDataGenerator

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "data", "ml-1m", "processed")

# Official ByteDance HLLM default configurations
DEFAULT_CONFIG = {
    'MAX_ITEM_LIST_LENGTH': 50,
    'MAX_TEXT_LENGTH': 256,
    'item_emb_token_n': 1,
    'loss': 'nce',
    'num_negatives': 512,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'epochs': 5,
}


def check_training_environment(device, model_type, dataset_path):
    """Check GPU, CUDA, VRAM, and required files for training.

    Args:
        device (str): Device to use ('cuda' or 'cpu').
        model_type (str): Type of LLM ('tinyllama' or 'baichuan2').
        dataset_path (str): Path to dataset directory.

    Returns:
        bool: True if environment is suitable, False otherwise.
    """
    print("\n" + "=" * 80)
    print("训练环境检查")
    print("=" * 80)

    # Check CUDA availability
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ 错误：指定了 --device cuda，但系统中没有可用的 GPU")
            print("   请检查 GPU 驱动程序或使用 --device cpu")
            return False

        print("✅ GPU 可用")
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")

        # Check VRAM
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   总显存: {total_memory:.2f} GB")

        print("   显存取决于物品目录、编码器宽度和序列长度，不能只按卡容量保证。")
        print("   小显存可尝试 --d_model 256 --batch_size 8 --max_seq_len 50 --loss_type sampled_softmax --amp")
    else:
        print("✅ 使用 CPU 进行训练（速度会很慢）")

    # Check item embeddings file
    emb_file = os.path.join(dataset_path, f'item_embeddings_{model_type}.pt')
    if not os.path.exists(emb_file):
        print(f"\n❌ 错误：Item embeddings 文件不存在: {emb_file}")
        print("   请先运行以下命令生成 embeddings：")
        print("   cd examples/generative/data/ml-1m")
        print(f"   python preprocess_hllm_data.py --model_type {model_type} --device {device}")
        return False

    print("✅ Item embeddings 文件存在")

    # Check data files
    required_files = ['vocab.pkl', 'train_data.pkl', 'val_data.pkl', 'test_data.pkl']
    for fname in required_files:
        fpath = os.path.join(dataset_path, fname)
        if not os.path.exists(fpath):
            print(f"\n❌ 错误：数据文件不存在: {fpath}")
            print("   请先运行以下命令预处理数据：")
            print("   cd examples/generative/data/ml-1m")
            print("   python preprocess_ml_hstu.py")
            return False

    print("✅ 所有数据文件存在")
    print("✅ 环境检查通过\n")
    return True


def get_movielens_data(data_dir=None):
    """Load preprocessed MovieLens-1M data from disk.

    Args:
        data_dir (str): Directory containing preprocessed files.
                       If None, uses default path relative to script location.

    Returns:
        tuple: ``(train_data, val_data, test_data, vocab_size)``.
    """
    print("加载真实数据...")

    # Use default path if not provided
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    # Convert to absolute path
    data_dir = os.path.abspath(data_dir)

    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请先运行: python examples/generative/data/ml-1m/preprocess_ml_hstu.py")
        return None

    # 加载词表
    vocab_file = os.path.join(data_dir, 'vocab.pkl')
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    item_to_idx = vocab['item_to_idx'] if 'item_to_idx' in vocab else vocab
    vocab_size = max(item_to_idx.values()) + 1

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


def load_item_embeddings(data_dir, model_type='tinyllama'):
    """Load pre-computed item embeddings.
    
    Args:
        data_dir (str): Directory containing item_embeddings.pt.
        model_type (str): Type of LLM used ('tinyllama' or 'baichuan2').
        
    Returns:
        Tensor: Item embeddings of shape (vocab_size, d_model).
    """
    emb_file = os.path.join(data_dir, f'item_embeddings_{model_type}.pt')

    if not os.path.exists(emb_file):
        print(f"❌ Item embeddings文件不存在: {emb_file}")
        print(f"请先运行: python examples/generative/data/ml-1m/preprocess_hllm_data.py --model_type {model_type}")
        return None

    embeddings = torch.load(emb_file)
    print(f"✅ 加载item embeddings: {embeddings.shape}")
    return embeddings


def main(dataset_path, model_type, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, max_seq_len, loss_type='nce', d_model=None, num_negatives=128, accumulation_steps=1, amp=False, learn_temperature=False):
    """Main training and evaluation entry point."""
    torch.manual_seed(seed)

    print("=" * 80)
    print(f"HLLM Model Example on MovieLens ({model_type})")
    print("=" * 80)

    # Convert to absolute path and display
    dataset_path = os.path.abspath(dataset_path)
    print(f"\n📂 数据目录: {dataset_path}\n")

    # Check training environment
    if not check_training_environment(device, model_type, dataset_path):
        sys.exit(1)

    # Load data
    print("\nLoading data...")
    result = get_movielens_data(dataset_path)
    if result is None:
        return

    train_data, val_data, test_data, vocab_size = result

    # MovieLens preprocessing left-pads histories; retain the most recent events.
    for data in [train_data, val_data, test_data]:
        for key in ['seq_tokens', 'seq_positions', 'seq_time_diffs']:
            data[key] = np.asarray(data[key])[:, -max_seq_len:]

    # Load item embeddings
    print("\nLoading item embeddings...")
    item_embeddings = load_item_embeddings(dataset_path, model_type)
    if item_embeddings is None:
        return

    # Auto-detect d_model from item embeddings dimension
    d_model = item_embeddings.shape[1] if d_model is None else d_model
    print(f"✅ 缓存 embedding 维度: {item_embeddings.shape[1]}，User 编码器维度: {d_model}")

    # Model hyper-parameters
    # Adjust n_heads based on d_model to ensure d_model % n_heads == 0
    if d_model >= 2048:
        n_heads = 16  # For TinyLlama (2048) or Baichuan2 (4096)
    elif d_model >= 512:
        n_heads = 8
    else:
        n_heads = 4

    n_layers = 2

    print(f"Device: {device}")
    print(f"Hidden dim: {d_model}")
    print(f"Num heads: {n_heads}")
    print(f"Num layers: {n_layers}")
    print(f"Max sequence length: {max_seq_len}")

    # Build data loaders
    print("\nBuilding data loaders...")
    train_gen = SequenceDataGenerator(train_data['seq_tokens'], train_data['seq_positions'], train_data['targets'], train_data['seq_time_diffs'])
    val_gen = SequenceDataGenerator(val_data['seq_tokens'], val_data['seq_positions'], val_data['targets'], val_data['seq_time_diffs'])
    test_gen = SequenceDataGenerator(test_data['seq_tokens'], test_data['seq_positions'], test_data['targets'], test_data['seq_time_diffs'])

    train_dataloader = train_gen.generate_dataloader(batch_size=batch_size, num_workers=0)[0]
    val_dataloader = val_gen.generate_dataloader(batch_size=batch_size, num_workers=0, shuffle=False)[0]
    test_dataloader = test_gen.generate_dataloader(batch_size=batch_size, num_workers=0, shuffle=False)[0]

    print(f"Train size: {len(train_dataloader.dataset)}")
    print(f"Val size: {len(val_dataloader.dataset)}")
    print(f"Test size: {len(test_dataloader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = HLLMModel(
        item_embeddings=item_embeddings,
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=0.1,
        project_items=True,
        learn_temperature=learn_temperature,
    )
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    print("\nCreating trainer...")
    # Configure loss function parameters
    if loss_type == 'sampled_softmax':
        loss_params = {'num_negatives': num_negatives, 'negative_similarity_threshold': 0.99}
    elif loss_type == 'nce':
        loss_params = {"temperature": 1.0, "ignore_index": 0}
    else:
        loss_params = {"ignore_index": 0}

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
        loss_type=loss_type,
        loss_params=loss_params,
        accumulation_steps=accumulation_steps,
        amp=amp,
    )
    print(f"✅ 使用 {loss_type.upper()} Loss 函数")

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
    predictions = trainer.predict(test_dataloader, top_k=max(topKs))
    y_true = {str(i): [int(target)] for i, target in enumerate(test_data['targets'])}
    y_pred = {str(i): items.tolist() for i, items in enumerate(predictions)}
    ranking_results = topk_metrics(y_true, y_pred, topKs=topKs)

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
    parser = argparse.ArgumentParser(
        description='HLLM Model Example on MovieLens Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small encoder with frozen Item LLM embeddings (not a pretrained User LLM)
  python examples/generative/run_hllm_movielens.py --d_model 256 --batch_size 8 --max_seq_len 50 --loss_type sampled_softmax --amp

  # From project root directory
  python examples/generative/run_hllm_movielens.py --model_type tinyllama --device cuda

  # With custom dataset path
  python examples/generative/run_hllm_movielens.py --dataset_path ./examples/generative/data/ml-1m/processed/ --model_type tinyllama --device cuda
        """
    )
    parser.add_argument('--dataset_path', default=None, help='Path to dataset directory (default: auto-detect from script location)')
    parser.add_argument('--model_type', default='tinyllama', choices=['tinyllama', 'baichuan2'], help='Type of LLM to use')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--save_dir', default='./', help='Directory to save model')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--loss_type', default='nce', choices=['cross_entropy', 'nce', 'sampled_softmax'], help='Legacy nce uses full vocabulary; sampled_softmax uses sampled negatives')
    parser.add_argument('--d_model', type=int, default=None, help='Optional small encoder width, e.g. 256; default matches item embeddings')
    parser.add_argument('--num_negatives', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--learn_temperature', action='store_true')

    args = parser.parse_args()

    # Use default path if not provided
    dataset_path = args.dataset_path if args.dataset_path else _DEFAULT_DATA_DIR

    main(
        dataset_path,
        args.model_type,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.device,
        args.save_dir,
        args.seed,
        args.max_seq_len,
        args.loss_type,
        args.d_model,
        args.num_negatives,
        args.accumulation_steps,
        args.amp,
        args.learn_temperature
    )
