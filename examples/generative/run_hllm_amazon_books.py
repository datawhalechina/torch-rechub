"""HLLM Model Example on Amazon Books Dataset.

This is the default dataset for HLLM, following the ByteDance official implementation.
"""

import os
import pickle
import sys

import numpy as np
import torch
import tqdm

from torch_rechub.basic.metric import topk_metrics
from torch_rechub.models.generative.hllm import HLLMModel
from torch_rechub.trainers.seq_trainer import SeqTrainer
from torch_rechub.utils.data import SequenceDataGenerator

sys.path.append("../..")

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "data", "amazon-books", "processed")


def check_training_environment(device, model_type, dataset_path):
    """Check GPU, CUDA, VRAM, and required files for training."""
    print("\n" + "=" * 80)
    print("Training Environment Check")
    print("=" * 80)

    # Check CUDA availability
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ Error: GPU not available but --device cuda specified")
            return False

        print("✅ GPU available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

        # Check VRAM
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   Total VRAM: {total_memory:.2f} GB")

        required_vram = 6
        if total_memory < required_vram:
            print(f"⚠️  Warning: HLLM training requires at least {required_vram}GB VRAM")
            response = input("   Continue? (y/n): ").strip().lower()
            if response != 'y':
                return False
        else:
            print("✅ VRAM sufficient")
    else:
        print("✅ Using CPU (will be slow)")

    # Check item embeddings file
    emb_file = os.path.join(dataset_path, f'item_embeddings_{model_type}.pt')
    if not os.path.exists(emb_file):
        print(f"\n❌ Error: Item embeddings file not found: {emb_file}")
        print("   Please run preprocessing first:")
        print("   cd examples/generative/data/amazon-books")
        print(f"   python preprocess_amazon_books_hllm.py --model_type {model_type} --device {device}")
        return False

    print("✅ Item embeddings file exists")

    # Check data files
    required_files = ['vocab.pkl', 'train_data.pkl', 'val_data.pkl', 'test_data.pkl']
    for fname in required_files:
        fpath = os.path.join(dataset_path, fname)
        if not os.path.exists(fpath):
            print(f"❌ Error: Required file not found: {fpath}")
            return False

    print("✅ All required data files exist")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HLLM training on Amazon Books dataset (Official)")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--model_type", default="tinyllama", choices=["tinyllama", "baichuan2"], help="LLM model type")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--loss_type", default="nce", choices=["cross_entropy", "nce"], help="Loss function type")

    args = parser.parse_args()

    # Check environment
    if not check_training_environment(args.device, args.model_type, args.data_dir):
        return

    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    # Load data
    with open(os.path.join(args.data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)

    with open(os.path.join(args.data_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    with open(os.path.join(args.data_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    with open(os.path.join(args.data_dir, 'item_text_map.pkl'), 'rb') as f:
        item_texts = pickle.load(f)

    vocab_size = len(vocab['item_to_idx'])
    print("✅ Data loaded")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Train samples: {len(train_data['targets'])}")
    print(f"   Val samples: {len(val_data['targets'])}")
    print(f"   Test samples: {len(test_data['targets'])}")

    # Load item embeddings
    emb_file = os.path.join(args.data_dir, f'item_embeddings_{args.model_type}.pt')
    item_embeddings = torch.load(emb_file, map_location='cpu')
    d_model = item_embeddings.shape[1]

    # Adjust n_heads based on d_model
    if d_model >= 2048:
        n_heads = 16
    elif d_model >= 512:
        n_heads = 8
    else:
        n_heads = 4

    print(f"   Item embeddings shape: {item_embeddings.shape}")
    print(f"   d_model: {d_model}, n_heads: {n_heads}")

    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    # Create model
    llm_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' if args.model_type == 'tinyllama' else 'baichuan-inc/Baichuan2-7B-Chat'

    model = HLLMModel(
        item_embeddings=None,
        vocab_size=len(item_texts),
        item_llm_path=llm_path,
        item_texts=item_texts,
        user_llm_path=llm_path,
        item_llm_8bit=True,
        user_llm_8bit=True,
        user_llm_gradient_checkpointing=True,
    )

    print("✅ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    # Configure loss function
    if args.loss_type == 'nce':
        loss_params = {"temperature": 0.1, "ignore_index": 0}
    else:
        loss_params = {"ignore_index": 0}

    trainer = SeqTrainer(
        model=model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': args.learning_rate, 'weight_decay': 1e-5},
        device=args.device,
        n_epoch=args.epochs,
        loss_type=args.loss_type,
        loss_params=loss_params,
    )
    print(f"✅ Using {args.loss_type.upper()} Loss")

    # Build data loaders
    print("\nBuilding data loaders...")
    train_gen = SequenceDataGenerator(
        train_data['seq_tokens'], train_data['seq_positions'],
        train_data['targets'], train_data['seq_time_diffs']
    )
    val_gen = SequenceDataGenerator(
        val_data['seq_tokens'], val_data['seq_positions'],
        val_data['targets'], val_data['seq_time_diffs']
    )

    train_dataloader = train_gen.generate_dataloader(batch_size=args.batch_size, num_workers=0)[0]
    val_dataloader = val_gen.generate_dataloader(batch_size=args.batch_size, num_workers=0)[0]

    print(f"Train size: {len(train_dataloader.dataset)}")
    print(f"Val size: {len(val_dataloader.dataset)}")

    # Train
    trainer.fit(train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    # Evaluate on test set
    model.eval()
    test_loader = SequenceDataGenerator(test_data, batch_size=args.batch_size, use_time_embedding=True)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
            seq_tokens = torch.LongTensor(batch['seq_tokens']).to(args.device)
            seq_time_diffs = torch.LongTensor(batch['seq_time_diffs']).to(args.device)
            targets = batch['targets']

            logits = model(seq_tokens, seq_time_diffs)
            preds = logits[:, -1, :].cpu().numpy()

            all_preds.append(preds)
            all_targets.extend(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.array(all_targets)

    # Calculate metrics
    metrics = topk_metrics(all_targets, all_preds, topKs=[10, 50, 200])
    print("\n✅ Test Results:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
