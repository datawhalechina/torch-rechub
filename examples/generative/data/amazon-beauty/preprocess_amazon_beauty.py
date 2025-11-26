"""Generate HSTU format sequence data from Amazon Beauty dataset.

This script processes the Amazon Beauty dataset and generates sequence data
in HSTU format (seq_tokens, seq_positions, seq_time_diffs, targets).

The dataset should be downloaded from: http://jmcauley.ucsd.edu/data/amazon/

Expected files:
    - reviews_Beauty_5.json: User reviews with timestamps
    - meta_Beauty.json: Product metadata

Output:
    - vocab.pkl: Product ID vocabulary
    - train_data.pkl: Training sequences
    - val_data.pkl: Validation sequences
    - test_data.pkl: Test sequences
"""

import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from download_utils import ensure_file_exists
from tqdm import tqdm

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = _SCRIPT_DIR
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "processed")

# Amazon dataset URLs (multiple sources for reliability)
# Note: Official sources require form submission, alternatives are provided
_REVIEWS_URLS = [
    # Official source (requires form at https://nijianmo.github.io/amazon/index.html)
    "https://nijianmo.github.io/amazon/index.html",
    # Alternative sources (no form required)
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023",
    "https://www.kaggle.com/datasets/wajahat1064/amazon-reviews-data-2023"
]

_META_URLS = [
    # Official source (requires form at https://nijianmo.github.io/amazon/index.html)
    "https://nijianmo.github.io/amazon/index.html",
    # Alternative sources (no form required)
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023",
    "https://www.kaggle.com/datasets/wajahat1064/amazon-reviews-data-2023"
]


def load_reviews(data_dir):
    """Load reviews from reviews_Beauty_5.json.

    Automatically downloads the file if it doesn't exist.
    """
    reviews_file = os.path.join(data_dir, "reviews_Beauty_5.json")

    # Ensure file exists (download if necessary)
    reviews_file = ensure_file_exists("reviews_Beauty_5.json", _REVIEWS_URLS, data_dir, auto_download=True)

    if reviews_file is None:
        raise FileNotFoundError(f"Reviews file not found and download failed: {os.path.join(data_dir, 'reviews_Beauty_5.json')}")

    print(f"\n📖 Loading reviews from {reviews_file}...")

    reviews = []
    with open(reviews_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                review = json.loads(line)
                reviews.append(review)
            except json.JSONDecodeError:
                continue

    return reviews


def build_user_sequences(reviews, min_seq_len=2):
    """Build user interaction sequences sorted by timestamp."""
    user_sequences = defaultdict(list)

    print("Building user sequences...")
    for review in tqdm(reviews, desc="Processing reviews"):
        user_id = review.get('reviewerID')
        product_id = review.get('asin')
        timestamp = review.get('unixReviewTime', 0)

        if user_id and product_id and timestamp:
            user_sequences[user_id].append({'product_id': product_id, 'timestamp': timestamp})

    # Sort by timestamp and filter by minimum sequence length
    valid_sequences = {}
    for user_id, interactions in user_sequences.items():
        interactions.sort(key=lambda x: x['timestamp'])
        if len(interactions) >= min_seq_len:
            valid_sequences[user_id] = interactions

    print(f"Found {len(valid_sequences)} users with >= {min_seq_len} interactions")
    return valid_sequences


def build_vocab(user_sequences):
    """Build product ID vocabulary."""
    product_ids = set()
    for interactions in user_sequences.values():
        for interaction in interactions:
            product_ids.add(interaction['product_id'])

    vocab = {pid: idx for idx, pid in enumerate(sorted(product_ids))}
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def generate_sequences(user_sequences, vocab, max_seq_len=200):
    """Generate training sequences."""
    sequences = []

    print("Generating sequences...")
    for user_id, interactions in tqdm(user_sequences.items(), desc="Generating sequences"):
        if len(interactions) < 2:
            continue

        # Generate sequences with sliding window
        for i in range(1, len(interactions)):
            seq_len = min(i, max_seq_len)
            start_idx = max(0, i - seq_len)

            seq_interactions = interactions[start_idx:i + 1]
            seq_tokens = [vocab[inter['product_id']] for inter in seq_interactions[:-1]]
            target = vocab[seq_interactions[-1]['product_id']]

            # Calculate time differences (in seconds)
            timestamps = [inter['timestamp'] for inter in seq_interactions]
            query_time = timestamps[-1]
            time_diffs = [query_time - ts for ts in timestamps[:-1]]

            # Calculate positions
            positions = list(range(len(seq_tokens)))

            sequences.append({'seq_tokens': seq_tokens, 'seq_positions': positions, 'seq_time_diffs': time_diffs, 'target': target})

    print(f"Generated {len(sequences)} sequences")
    return sequences


def split_data(sequences, train_ratio=0.8, val_ratio=0.1):
    """Split sequences into train/val/test sets.

    Returns data in the same format as MovieLens preprocessing:
    - Dictionary with keys: 'seq_tokens', 'seq_positions', 'seq_time_diffs', 'targets'
    - Each value is a numpy array
    """
    import numpy as np

    n = len(sequences)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_seqs = sequences[:train_size]
    val_seqs = sequences[train_size:train_size + val_size]
    test_seqs = sequences[train_size + val_size:]

    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    def convert_to_dict_format(seqs):
        """Convert list of sequence dicts to dict of arrays format."""
        # Pad sequences to same length
        max_len = max(len(seq['seq_tokens']) for seq in seqs) if seqs else 0

        seq_tokens_list = []
        seq_positions_list = []
        seq_time_diffs_list = []
        targets_list = []

        for seq in seqs:
            tokens = seq['seq_tokens']
            positions = seq['seq_positions']
            time_diffs = seq['seq_time_diffs']
            target = seq['target']

            # Pad to max_len
            pad_len = max_len - len(tokens)
            padded_tokens = [0] * pad_len + tokens  # Pad at the beginning
            padded_positions = list(range(pad_len)) + positions  # Adjust positions
            padded_time_diffs = [0] * pad_len + time_diffs  # Pad time diffs

            seq_tokens_list.append(padded_tokens)
            seq_positions_list.append(padded_positions)
            seq_time_diffs_list.append(padded_time_diffs)
            targets_list.append(target)

        return {
            'seq_tokens': np.array(seq_tokens_list,
                                   dtype=np.int64),
            'seq_positions': np.array(seq_positions_list,
                                      dtype=np.int64),
            'seq_time_diffs': np.array(seq_time_diffs_list,
                                       dtype=np.float32),
            'targets': np.array(targets_list,
                                dtype=np.int64)
        }

    train_data = convert_to_dict_format(train_seqs)
    val_data = convert_to_dict_format(val_seqs)
    test_data = convert_to_dict_format(test_seqs)

    return train_data, val_data, test_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Amazon Beauty dataset for HSTU")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--min_seq_len", type=int, default=2, help="Minimum sequence length")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process data
    reviews = load_reviews(args.data_dir)
    user_sequences = build_user_sequences(reviews, min_seq_len=args.min_seq_len)
    vocab = build_vocab(user_sequences)
    sequences = generate_sequences(user_sequences, vocab, max_seq_len=args.max_seq_len)
    train_data, val_data, test_data = split_data(sequences)

    # Save outputs
    print("\nSaving outputs...")
    with open(os.path.join(args.output_dir, "vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)

    with open(os.path.join(args.output_dir, "train_data.pkl"), 'wb') as f:
        pickle.dump(train_data, f)

    with open(os.path.join(args.output_dir, "val_data.pkl"), 'wb') as f:
        pickle.dump(val_data, f)

    with open(os.path.join(args.output_dir, "test_data.pkl"), 'wb') as f:
        pickle.dump(test_data, f)

    print(f"✅ Preprocessing complete!")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Vocab size: {len(vocab)}")
    print(f"   Total sequences: {len(sequences)}")


if __name__ == "__main__":
    main()
