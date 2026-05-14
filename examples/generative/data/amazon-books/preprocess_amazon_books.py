"""Amazon Books data preprocessing script for HSTU format.

This script processes Amazon Books interactions into HSTU-compatible format:
1. Load and filter interactions (users and items with >= 5 interactions)
2. Generate user sequences sorted by timestamp
3. Split into train/val/test sets
4. Save preprocessed data files

Supported data sources:
- raw: Stanford SNAP ratings_Books.csv
- bytedance: ByteDance HLLM processed Interactions/amazon_books.csv

Usage:
    python preprocess_amazon_books.py --data_source bytedance
    python preprocess_amazon_books.py --data_source raw
    python preprocess_amazon_books.py --no_download
"""

import os
import pickle
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = _SCRIPT_DIR
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "processed")

INTERACTION_SOURCES = {
    "raw": {
        "url": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
        "filename": "ratings_Books.csv",
        "description": "Stanford SNAP raw ratings",
    },
    "bytedance": {
        "url": "https://huggingface.co/ByteDance/HLLM/resolve/main/Interactions/amazon_books.csv",
        "filename": "ratings_Books.csv",
        "description": "ByteDance HLLM processed interactions",
    },
}
RAW_INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
BYTEDANCE_INTERACTION_COLUMNS = ["item_id", "user_id", "timestamp"]
REQUIRED_INTERACTION_COLUMNS = ["user_id", "item_id", "timestamp"]


def download_file(url, output_path, overwrite=False):
    """Download a file to ``output_path``.

    Existing files are kept by default. Pass ``overwrite=True`` to refresh a
    previously downloaded file.
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"File exists, skip download: {output_path}")
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"

    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")
    progress_bar = None

    def _progress_hook(block_num, block_size, total_size):
        nonlocal progress_bar
        if progress_bar is None:
            total = total_size if total_size > 0 else None
            progress_bar = tqdm.tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(output_path),
            )

        downloaded = block_num * block_size
        if progress_bar.total is not None:
            downloaded = min(downloaded, progress_bar.total)
        progress_bar.update(downloaded - progress_bar.n)

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress_hook)
        os.replace(tmp_path, output_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"Download failed: {exc}")
        return False
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(f"Downloaded: {output_path}")
    return True


def _csv_has_columns(file_path, expected_columns):
    try:
        first_line = pd.read_csv(file_path, nrows=0).columns.tolist()
    except Exception:
        return False
    return first_line == expected_columns


def _is_supported_interaction_csv(file_path):
    return (_csv_has_columns(file_path, RAW_INTERACTION_COLUMNS) or _csv_has_columns(file_path, BYTEDANCE_INTERACTION_COLUMNS))


def get_ratings_file(data_dir, data_source):
    """Return the interaction file path for the selected source."""
    source = INTERACTION_SOURCES[data_source]
    primary_path = os.path.join(data_dir, source["filename"])
    if os.path.exists(primary_path):
        return primary_path

    # Manual ByteDance downloads often keep the original amazon_books.csv name.
    if data_source == "bytedance":
        fallback_names = ["amazon_books_interactions.csv", "amazon_books.csv"]
        for name in fallback_names:
            fallback_path = os.path.join(data_dir, name)
            if os.path.exists(fallback_path) and _is_supported_interaction_csv(fallback_path):
                return fallback_path

    return primary_path


def read_interactions(file_path):
    """Read Amazon Books interactions and keep columns required downstream.

    Supported formats:
    - Raw SNAP: no header, ``user_id,item_id,rating,timestamp``
    - Standard CSV: header ``user_id,item_id,rating,timestamp``
    - ByteDance processed CSV: header ``item_id,user_id,timestamp``
    """
    header_columns = pd.read_csv(file_path, nrows=0).columns.tolist()

    if header_columns == RAW_INTERACTION_COLUMNS:
        ratings = pd.read_csv(file_path)
    elif header_columns == BYTEDANCE_INTERACTION_COLUMNS:
        ratings = pd.read_csv(file_path)
        ratings = ratings[REQUIRED_INTERACTION_COLUMNS]
    else:
        ratings = pd.read_csv(file_path, sep=",", names=RAW_INTERACTION_COLUMNS, header=None)

    ratings = ratings[REQUIRED_INTERACTION_COLUMNS]
    ratings["timestamp"] = ratings["timestamp"].astype(float)
    return ratings


def prepare_ratings_file(data_dir, data_source, download=True, overwrite=False):
    """Download the selected interaction file when requested."""
    source = INTERACTION_SOURCES[data_source]
    output_path = os.path.join(data_dir, source["filename"])

    if not download:
        print("Download disabled; using existing interaction file if available.")
        return True

    print(f"Selected interaction source: {data_source} ({source['description']})")
    return download_file(source["url"], output_path, overwrite=overwrite)


def load_ratings(data_dir, data_source, filter_interactions=False, min_interactions=5):
    """Load and preprocess Amazon Books ratings.

    For raw data, filtering users and items by interaction count is useful to
    remove sparse cold-start rows. For ByteDance processed data, the default
    workflow keeps the provided interaction file unchanged.
    """
    ratings_file = get_ratings_file(data_dir, data_source)

    if not os.path.exists(ratings_file):
        print(f"❌ Error: Ratings file not found: {ratings_file}")
        print("\nRun with download enabled, or place an existing file in data_dir:")
        print("  python preprocess_amazon_books.py --data_source bytedance")
        print("  python preprocess_amazon_books.py --data_source raw")
        print("  python preprocess_amazon_books.py --no_download")
        return None

    print(f"\n📖 Loading ratings from {ratings_file}...")

    ratings = read_interactions(ratings_file)

    print(f"  Raw data: {len(ratings)} interactions")
    print(f"  Users: {ratings['user_id'].nunique()}")
    print(f"  Items: {ratings['item_id'].nunique()}")

    if not filter_interactions:
        print("\n📊 Skipping interaction-count filtering")
        return ratings

    # Filter sparse users/items.
    print(f"\n📊 Filtering (>={min_interactions} interactions)...")

    item_counts = ratings['item_id'].value_counts()
    user_counts = ratings['user_id'].value_counts()

    valid_items = item_counts[item_counts >= min_interactions].index
    valid_users = user_counts[user_counts >= min_interactions].index

    ratings = ratings[ratings['item_id'].isin(valid_items)]
    ratings = ratings[ratings['user_id'].isin(valid_users)]

    # Additional filter: ensure each user still has enough items after item filtering.
    ratings = ratings.groupby('user_id').filter(lambda x: len(x) >= min_interactions)

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
    parser.add_argument("--data_source", default="bytedance", choices=["raw", "bytedance"], help="Interaction data source to use")
    parser.add_argument("--no_download", action="store_true", help="Skip download and use an existing interaction file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing downloaded file")
    parser.add_argument("--min_interactions", type=int, default=5, help="Minimum interactions for raw data filtering")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--min_seq_len", type=int, default=5, help="Minimum sequence length")

    args = parser.parse_args()
    filter_interactions = args.data_source == "raw"

    print("=" * 80)
    print("Amazon Books Data Preprocessing (HSTU Format)")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data source: {args.data_source}")
    print(f"Download: {not args.no_download}")
    print(f"Overwrite downloads: {args.overwrite}")
    print(f"Filter interactions: {filter_interactions}")
    print(f"Minimum interactions: {args.min_interactions}")

    # Step 0: Download or reuse interaction data.
    prepare_ratings_file(
        args.data_dir,
        args.data_source,
        download=not args.no_download,
        overwrite=args.overwrite,
    )

    # Step 1: Load ratings
    ratings = load_ratings(args.data_dir, args.data_source, filter_interactions=filter_interactions, min_interactions=args.min_interactions)
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
