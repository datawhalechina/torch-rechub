"""MovieLens-1M preprocessing script for the HSTU model.

This script converts the raw MovieLens-1M data into the format required by
HSTU-based generative recommendation experiments:

- build interaction sequences per user;
- sort interactions by timestamp;
- generate multiple training samples per user via a sliding window;
- create train/validation/test splits at the user level;
- output ``seq_tokens``, ``seq_positions``, ``seq_time_diffs`` and ``targets``.

Reference implementation: https://github.com/meta-recsys/generative-recommenders
Key difference: we use an explicit sliding-window strategy to greatly increase
training data while preserving temporal order.
"""

import os
import pickle
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_ARCHIVE = "ml-1m.zip"
MOVIELENS_REQUIRED_FILES = ("ratings.dat", "movies.dat", "users.dat")


def download_file(url, output_path, overwrite=False):
    """Download a file with a progress bar."""
    if os.path.exists(output_path) and not overwrite:
        print(f"文件已存在，跳过下载: {output_path}")
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    progress_bar = None

    print(f"下载: {url}")
    print(f"保存到: {output_path}")

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
        print(f"下载失败: {exc}")
        return False
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(f"下载完成: {output_path}")
    return True


def extract_movielens_archive(archive_path, data_dir, overwrite=False):
    """Extract MovieLens-1M files from the official zip archive."""
    print(f"解压 MovieLens-1M 数据: {archive_path}")
    os.makedirs(data_dir, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        for filename in MOVIELENS_REQUIRED_FILES:
            member = f"ml-1m/{filename}"
            output_path = os.path.join(data_dir, filename)
            if member not in names:
                print(f"压缩包中缺少文件: {member}")
                return False
            if os.path.exists(output_path) and not overwrite:
                print(f"文件已存在，跳过解压: {output_path}")
                continue
            with archive.open(member) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
            print(f"已解压: {output_path}")

    return True


def ensure_movielens_data(data_dir, download=True, overwrite=False):
    """Download/extract MovieLens-1M data or validate existing local files."""
    required_paths = [os.path.join(data_dir, filename) for filename in MOVIELENS_REQUIRED_FILES]
    if not download:
        print("已禁用下载，将使用本地已有 MovieLens-1M 文件。")
        missing = [path for path in required_paths if not os.path.exists(path)]
        if missing:
            print("缺少必要文件:")
            for path in missing:
                print(f"  - {path}")
            return False
        return True

    archive_path = os.path.join(data_dir, MOVIELENS_ARCHIVE)
    if not download_file(MOVIELENS_1M_URL, archive_path, overwrite=overwrite):
        return False
    return extract_movielens_archive(archive_path, data_dir, overwrite=overwrite)


class MovieLensHSTUPreprocessor:
    """Preprocessor for MovieLens-1M data in HSTU format.

    This class applies a sliding-window strategy to each user's interaction
    sequence to generate multiple training samples, which significantly
    increases data utilization and improves model training.

    Example:
        User sequence: [item1, item2, item3, item4, item5]
        Generated samples:
            ([item1], item2)
            ([item1, item2], item3)
            ([item1, item2, item3], item4)
            ([item1, item2, item3, item4], item5)
    """

    def __init__(self, data_dir="./data/ml-1m/", output_dir="./data/ml-1m/processed/"):
        """Initialize the preprocessor.

        Args:
            data_dir (str): Directory containing the raw MovieLens-1M files.
            output_dir (str): Directory where processed files will be stored.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Core preprocessing hyperparameters
        self.min_seq_len = 5  # Minimum sequence length per user
        self.max_seq_len = 200  # Maximum sequence length (for truncation/padding)
        self.rating_threshold = 0  # Keep all MovieLens ratings by default.
        self.min_item_count = 1  # Keep all interacted items by default.

    def load_data(self):
        """Load raw MovieLens-1M rating and movie metadata files."""
        print("加载MovieLens-1M数据...")

        # 读取ratings数据
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(os.path.join(self.data_dir, 'ratings.dat'), sep='::', header=None, names=rnames, engine='python', encoding='ISO-8859-1')

        # 读取movies数据
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(os.path.join(self.data_dir, 'movies.dat'), sep='::', header=None, names=mnames, engine='python', encoding='ISO-8859-1')

        print(f"原始数据: {len(ratings)} 条评分, {ratings['user_id'].nunique()} 用户, {ratings['movie_id'].nunique()} 电影")

        return ratings, movies

    def filter_data(self, ratings):
        """Filter ratings by threshold and apply cold-start heuristics.

        The default procedure keeps all MovieLens interactions and all
        interacted items, matching the public HSTU/SASRec preprocessing setup.
        The threshold/count fields remain configurable in code for ablations.

        Args:
            ratings (pd.DataFrame): Raw ratings dataframe.

        Returns:
            pd.DataFrame: Filtered ratings dataframe.
        """
        print("\n数据过滤...")
        print(f"过滤前: {len(ratings)} 条评分")

        # 1) Filter out low ratings (keep only ratings >= threshold)
        ratings = ratings[ratings['rating'] >= self.rating_threshold]
        print(f"评分过滤后 (>={self.rating_threshold}): {len(ratings)} 条评分")

        # 2) Item cold-start filter: keep items with at least min_item_count ratings
        item_counts = ratings['movie_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_count].index
        ratings = ratings[ratings['movie_id'].isin(valid_items)]
        print(f"Item过滤后 (>={self.min_item_count}次): {len(ratings)} 条评分")

        # 3) User cold-start filter: keep users with at least min_seq_len interactions
        user_counts = ratings['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_seq_len].index
        ratings = ratings[ratings['user_id'].isin(valid_users)]
        print(f"User过滤后 (>={self.min_seq_len}次): {len(ratings)} 条评分")

        print(f"过滤后: {ratings['user_id'].nunique()} 用户, {ratings['movie_id'].nunique()} 电影")

        return ratings

    def create_vocab(self, ratings):
        """Create a structured mapping between raw movie IDs and token IDs.

        Args:
            ratings (pd.DataFrame): Filtered ratings dataframe.

        Returns:
            dict: Vocabulary with ``item_to_idx`` and ``idx_to_item`` mappings.
        """
        print("\n创建词表...")

        unique_movies = sorted(ratings['movie_id'].unique())
        # token_id starts from 1; 0 is reserved for PAD
        item_to_idx = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movies)}
        idx_to_item = {idx: movie_id for movie_id, idx in item_to_idx.items()}
        vocab = {'item_to_idx': item_to_idx, 'idx_to_item': idx_to_item}

        print(f"词表大小: {len(item_to_idx) + 1} (包含PAD)")
        return vocab

    def build_user_sequences(self, ratings):
        """Build sorted user interaction sequences.

        For each user, interactions are sorted by timestamp and both the
        movie_ids and timestamps are stored.

        Args:
            ratings (pd.DataFrame): Filtered ratings dataframe.

        Returns:
            tuple: ``(user_sequences, user_timestamps)`` where
                - ``user_sequences`` is ``{user_id: [movie_id1, ...]}``
                - ``user_timestamps`` is ``{user_id: [timestamp1, ...]}``
        """
        print("\n构建用户序列...")

        # Sort by user and timestamp
        ratings_sorted = ratings.sort_values(['user_id', 'timestamp'])

        # Group by user and collect movie_ids and timestamps
        user_sequences = {}
        user_timestamps = {}
        for user_id, group in ratings_sorted.groupby('user_id'):
            user_sequences[user_id] = group['movie_id'].tolist()
            user_timestamps[user_id] = group['timestamp'].tolist()

        # Log sequence length statistics
        seq_lengths = [len(seq) for seq in user_sequences.values()]
        print(f"用户数: {len(user_sequences)}")
        print(f"序列长度统计: 平均={np.mean(seq_lengths):.1f}, 最小={np.min(seq_lengths)}, 最大={np.max(seq_lengths)}")

        return user_sequences, user_timestamps

    def generate_leave_last_out_samples(self, user_sequences, user_timestamps, vocab):
        """Generate train/val/test samples using **leave-last-out per user**.

        Standard SASRec / HSTU evaluation protocol:

        - Test sample (per user): ``input = items[:-1]``, ``target = items[-1]``.
        - Val sample (per user): ``input = items[:-2]``, ``target = items[-2]``.
        - Train samples (per user): sliding window over the remaining prefix.
          For ``i`` in ``[1, N-3]``, sample = ``(items[:i], items[i])``. This
          gives ``N - 3`` training samples per user without leaking the val or
          test targets.

        Time diffs follow the Meta reference: per-position seconds delta from
        the prefix's last (query) timestamp, padded on the left to
        ``max_seq_len``.

        Returns
        -------
        dict
            ``{'train', 'val', 'test'}`` each containing numpy arrays for
            ``seq_tokens``, ``seq_positions``, ``seq_time_diffs``, ``targets``.
        """
        print("\n按留一法生成 train/val/test 样本（滑动窗口 + 时间戳）...")
        item_to_idx = vocab['item_to_idx'] if 'item_to_idx' in vocab else vocab

        splits = {
            'train': {
                'seq_tokens': [],
                'seq_positions': [],
                'seq_time_diffs': [],
                'targets': []
            },
            'val': {
                'seq_tokens': [],
                'seq_positions': [],
                'seq_time_diffs': [],
                'targets': []
            },
            'test': {
                'seq_tokens': [],
                'seq_positions': [],
                'seq_time_diffs': [],
                'targets': []
            },
        }

        positions_template = list(range(self.max_seq_len))

        def _append(bucket, history, history_ts, target):
            seq_tokens = [item_to_idx.get(m, 0) for m in history]
            target_token = item_to_idx.get(target, 0)

            # Time diffs relative to the prefix's last timestamp (query time),
            # matching the Meta reference `query_time - timestamps`.
            query_ts = history_ts[-1]
            seq_time_diffs = [query_ts - t for t in history_ts]

            # Truncate to max_seq_len keeping the most recent history.
            if len(seq_tokens) > self.max_seq_len:
                seq_tokens = seq_tokens[-self.max_seq_len:]
                seq_time_diffs = seq_time_diffs[-self.max_seq_len:]

            # Left-pad to max_seq_len with PAD=0 / time_diff=0.
            pad_len = self.max_seq_len - len(seq_tokens)
            seq_tokens = [0] * pad_len + seq_tokens
            seq_time_diffs = [0] * pad_len + seq_time_diffs

            bucket['seq_tokens'].append(seq_tokens)
            bucket['seq_positions'].append(positions_template)
            bucket['seq_time_diffs'].append(seq_time_diffs)
            bucket['targets'].append(target_token)

        kept_users = 0
        skipped_users = 0
        for user_id, sequence in user_sequences.items():
            timestamps = user_timestamps[user_id]
            N = len(sequence)
            # Need at least 4 items for 1 train + 1 val + 1 test sample.
            if N < 4:
                skipped_users += 1
                continue
            kept_users += 1

            # Train: prefixes ending at items[1..N-3], excluding val/test targets.
            for end_idx in range(1, N - 2):
                _append(splits['train'], sequence[:end_idx], timestamps[:end_idx], sequence[end_idx])

            # Val: input items[:N-2], target items[N-2].
            _append(splits['val'], sequence[:N - 2], timestamps[:N - 2], sequence[N - 2])

            # Test: input items[:N-1], target items[N-1].
            _append(splits['test'], sequence[:N - 1], timestamps[:N - 1], sequence[N - 1])

        for name in splits:
            for key in ('seq_tokens', 'seq_positions', 'seq_time_diffs', 'targets'):
                dtype = np.int64 if key == 'seq_time_diffs' else np.int32
                splits[name][key] = np.array(splits[name][key], dtype=dtype)

        # Time-difference statistics over training prefixes.
        train_td = splits['train']['seq_time_diffs']
        non_zero_td = train_td[train_td > 0]
        if non_zero_td.size > 0:
            print("时间差统计（秒，train 集）:")
            print(f"  平均: {np.mean(non_zero_td):.1f}")
            print(f"  中位数: {np.median(non_zero_td):.1f}")
            print(f"  最小: {np.min(non_zero_td)}")
            print(f"  最大: {np.max(non_zero_td)}")
            print(f"  平均（小时）: {np.mean(non_zero_td) / 3600:.1f}")
            print(f"  平均（天）: {np.mean(non_zero_td) / 86400:.1f}")

        print(f"保留用户: {kept_users} | 跳过 (<4 items): {skipped_users}")
        print(f"样本数: train={len(splits['train']['targets'])}, "
              f"val={len(splits['val']['targets'])}, test={len(splits['test']['targets'])}")

        return splits

    def save_data(self, data_split, vocab):
        """Save processed data splits and vocabulary to disk.

        Args:
            data_split (dict): Dictionary containing train/val/test splits.
            vocab (dict): Vocabulary with ``item_to_idx`` and ``idx_to_item`` mappings.
        """
        print("\n保存数据...")

        # Save split data
        for split_name, split_data in data_split.items():
            output_file = os.path.join(self.output_dir, f'{split_name}_data.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"保存 {split_name} 数据到 {output_file}")
            print(f"  - seq_tokens: {split_data['seq_tokens'].shape}")
            print(f"  - seq_time_diffs: {split_data['seq_time_diffs'].shape}")  # 新增
            print(f"  - targets: {split_data['targets'].shape}")

        # 保存词表
        vocab_file = os.path.join(self.output_dir, 'vocab.pkl')
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"保存词表到 {vocab_file}")

    def preprocess(self):
        """Run the full preprocessing pipeline."""
        print("=" * 80)
        print("MovieLens-1M HSTU data preprocessing - sliding window with timestamps")
        print("=" * 80)

        # 1. 加载数据
        ratings, movies = self.load_data()

        # 2. 过滤数据
        ratings_filtered = self.filter_data(ratings)

        # 3. 创建词表
        vocab = self.create_vocab(ratings_filtered)

        # 4. 构建用户序列（包含时间戳）
        user_sequences, user_timestamps = self.build_user_sequences(ratings_filtered)

        # 5. 按留一法生成 train/val/test 样本
        data_split = self.generate_leave_last_out_samples(user_sequences, user_timestamps, vocab)

        # 6. 保存数据
        self.save_data(data_split, vocab)

        print("=" * 80)
        print("预处理完成！")
        print("=" * 80)
        print("\n关键改进:")
        print("✅ 采用留一法 (leave-last-out per user): 末位=test, 倒数第二=val, 余下做训练前缀")
        print("✅ 训练前缀采用滑动窗口，覆盖每个用户的全部上下文")
        print("✅ 添加冷启动过滤，提高数据质量")
        print("✅ 支持时间戳处理，计算时间差用于时间感知建模")


if __name__ == '__main__':
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="MovieLens-1M preprocessing for HSTU")
    parser.add_argument("--data_dir", default=script_dir, help="Directory containing MovieLens-1M raw files")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--no_download", action="store_true", help="Skip download and use existing local files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloaded/extracted files")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(data_dir, 'processed')

    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"下载: {not args.no_download}")
    print(f"覆盖已有文件: {args.overwrite}")

    if not ensure_movielens_data(data_dir, download=not args.no_download, overwrite=args.overwrite):
        raise SystemExit(1)

    preprocessor = MovieLensHSTUPreprocessor(data_dir=data_dir, output_dir=output_dir)
    preprocessor.preprocess()
