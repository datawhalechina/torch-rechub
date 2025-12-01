"""Amazon Books data preprocessing script for HSTU format.

This script processes Amazon Books dataset (ratings_Books.csv) into HSTU-compatible format:
1. Load and filter interactions (users and items with >= 5 interactions)
2. Generate user sequences sorted by timestamp
3. Split into train/val/test sets
4. Save preprocessed data files

Data format follows ByteDance HLLM official implementation:
- ratings_Books.csv: user_id, item_id, rating, timestamp

Usage:
    python preprocess_amazon_books.py --data_dir . --output_dir ./processed
"""

import gzip
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = _SCRIPT_DIR
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "processed")


def load_ratings(data_dir):
    """Load and preprocess Amazon Books ratings.

    Follows ByteDance HLLM official processing:
    - Filter users and items with >= 5 interactions
    """
    ratings_file = os.path.join(data_dir, "ratings_Books.csv")

    if not os.path.exists(ratings_file):
        print(f"❌ Error: Ratings file not found: {ratings_file}")
        print("\nPlease download the file from:")
        print("  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv")
        print("Or use the processed version from ByteDance:")
        print("  https://huggingface.co/ByteDance/HLLM/resolve/main/Interactions/amazon_books.csv")
        return None

    print(f"\n📖 Loading ratings from {ratings_file}...")

    # Load ratings (format: user_id, item_id, rating, timestamp)
    ratings = pd.read_csv(ratings_file, sep=",", names=["user_id", "item_id", "rating", "timestamp"], header=None)

    # Check if file has header
    if ratings.iloc[0]['user_id'] == 'user_id':
        ratings = ratings.iloc[1:]
        ratings['timestamp'] = ratings['timestamp'].astype(float)

    print(f"  Raw data: {len(ratings)} interactions")
    print(f"  Users: {ratings['user_id'].nunique()}")
    print(f"  Items: {ratings['item_id'].nunique()}")

    # Filter users and items with >= 5 interactions (following official implementation)
    print("\n📊 Filtering (>= 5 interactions)...")

    item_counts = ratings['item_id'].value_counts()
    user_counts = ratings['user_id'].value_counts()

    valid_items = item_counts[item_counts >= 5].index
    valid_users = user_counts[user_counts >= 5].index

    ratings = ratings[ratings['item_id'].isin(valid_items)]
    ratings = ratings[ratings['user_id'].isin(valid_users)]

    # Additional filter: ensure each user has >= 5 items after item filtering
    ratings = ratings.groupby('user_id').filter(lambda x: len(x) >= 5)

    print(f"  After filter: {len(ratings)} interactions")
    print(f"  Users: {ratings['user_id'].nunique()}")
    print(f"  Items: {ratings['item_id'].nunique()}")

    return ratings


def build_sequences(ratings, max_seq_len=200, min_seq_len=5):
    """Build user sequences from ratings, sorted by timestamp."""
    print(f"\n🔄 Building user sequences (max_len={max_seq_len}, min_len={min_seq_len})...")

    # Build vocabulary
    unique_items = ratings['item_id'].unique()
    item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}  # 0 reserved for padding
    item_to_idx['<PAD>'] = 0

    vocab = {'item_to_idx': item_to_idx, 'idx_to_item': {v: k for k, v in item_to_idx.items()}}

    print(f"  Vocabulary size: {len(item_to_idx)}")

    # Group by user and sort by timestamp
    user_sequences = defaultdict(list)

    for _, row in tqdm.tqdm(ratings.iterrows(), total=len(ratings), desc="Building sequences"):
        user_id = row['user_id']
        item_id = row['item_id']
        timestamp = float(row['timestamp'])

        item_idx = item_to_idx[item_id]
        user_sequences[user_id].append((timestamp, item_idx))

    # Sort each user's sequence by timestamp
    sequences = []
    for user_id, items in tqdm.tqdm(user_sequences.items(), desc="Sorting sequences"):
        items.sort(key=lambda x: x[0])  # Sort by timestamp

        if len(items) < min_seq_len:
            continue

        # Extract item indices and timestamps
        timestamps = [t for t, _ in items]
        item_indices = [idx for _, idx in items]

        # Truncate if too long
        if len(item_indices) > max_seq_len:
            item_indices = item_indices[-max_seq_len:]
            timestamps = timestamps[-max_seq_len:]

        sequences.append({'user_id': user_id, 'item_indices': item_indices, 'timestamps': timestamps})

    print(f"  Generated {len(sequences)} user sequences")

    return sequences, vocab


def split_data(sequences, train_ratio=0.8, val_ratio=0.1):
    """Split sequences into train/val/test sets using leave-one-out strategy."""
    print(f"\n✂️ Splitting data (train={train_ratio}, val={val_ratio})...")

    train_data = {'seq_tokens': [], 'seq_positions': [], 'seq_time_diffs': [], 'targets': []}
    val_data = {'seq_tokens': [], 'seq_positions': [], 'seq_time_diffs': [], 'targets': []}
    test_data = {'seq_tokens': [], 'seq_positions': [], 'seq_time_diffs': [], 'targets': []}

    for seq in tqdm.tqdm(sequences, desc="Splitting"):
        item_indices = seq['item_indices']
        timestamps = seq['timestamps']

        if len(item_indices) < 3:
            continue

        # Test: last item as target
        test_target = item_indices[-1]
        test_seq = item_indices[:-1]
        test_times = timestamps[:-1]

        # Validation: second-to-last item as target
        val_target = item_indices[-2]
        val_seq = item_indices[:-2]
        val_times = timestamps[:-2]

        # Train: all preceding items
        for i in range(2, len(item_indices) - 1):
            train_target = item_indices[i]
            train_seq = item_indices[:i]
            train_times = timestamps[:i]

            train_data['seq_tokens'].append(train_seq)
            train_data['seq_positions'].append(list(range(len(train_seq))))
            train_data['seq_time_diffs'].append([int(train_times[-1] - t) for t in train_times])
            train_data['targets'].append(train_target)

        # Add validation sample
        if len(val_seq) >= 2:
            val_data['seq_tokens'].append(val_seq)
            val_data['seq_positions'].append(list(range(len(val_seq))))
            val_data['seq_time_diffs'].append([int(val_times[-1] - t) for t in val_times])
            val_data['targets'].append(val_target)

        # Add test sample
        test_data['seq_tokens'].append(test_seq)
        test_data['seq_positions'].append(list(range(len(test_seq))))
        test_data['seq_time_diffs'].append([int(test_times[-1] - t) for t in test_times])
        test_data['targets'].append(test_target)

    print(f"  Train samples: {len(train_data['targets'])}")
    print(f"  Val samples: {len(val_data['targets'])}")
    print(f"  Test samples: {len(test_data['targets'])}")

    return train_data, val_data, test_data


def save_data(train_data, val_data, test_data, vocab, output_dir):
    """Save preprocessed data to files."""
    print(f"\n💾 Saving data to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Save vocabulary
    vocab_file = os.path.join(output_dir, 'vocab.pkl')
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"  ✅ Saved vocab.pkl ({len(vocab['item_to_idx'])} items)")

    # Save train/val/test data
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        file_path = os.path.join(output_dir, f'{name}_data.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✅ Saved {name}_data.pkl ({len(data['targets'])} samples)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Amazon Books data preprocessing for HSTU")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="Directory containing ratings_Books.csv")
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--min_seq_len", type=int, default=5, help="Minimum sequence length")

    args = parser.parse_args()

    print("=" * 80)
    print("Amazon Books Data Preprocessing (HSTU Format)")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Step 1: Load ratings
    ratings = load_ratings(args.data_dir)
    if ratings is None:
        return

    # Step 2: Build sequences
    sequences, vocab = build_sequences(ratings, args.max_seq_len, args.min_seq_len)

    # Step 3: Split data
    train_data, val_data, test_data = split_data(sequences)

    # Step 4: Save data
    save_data(train_data, val_data, test_data, vocab, args.output_dir)

    print("\n" + "=" * 80)
    print("✅ Preprocessing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
