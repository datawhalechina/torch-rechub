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
from collections import defaultdict

import numpy as np
import pandas as pd


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
        self.rating_threshold = 3  # Ratings >= threshold are treated as positive
        self.min_item_count = 5  # Item must appear at least this many times

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

        The procedure keeps only high-rated interactions, removes very rare
        items, and users with too few interactions.

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
        """Create a mapping from raw movie IDs to token IDs.

        Args:
            ratings (pd.DataFrame): Filtered ratings dataframe.

        Returns:
            dict: Mapping from ``movie_id`` to ``token_id``.
        """
        print("\n创建词表...")

        unique_movies = sorted(ratings['movie_id'].unique())
        # token_id starts from 1; 0 is reserved for PAD
        vocab = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movies)}
        vocab[0] = 0  # PAD token

        print(f"词表大小: {len(vocab)} (包含PAD)")
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

    def generate_training_samples_sliding_window(self, user_sequences, user_timestamps, vocab):
        """Generate training samples using a sliding window (with timestamps).

        Key ideas:
        * For each user sequence, generate multiple prefix-target pairs via a
          sliding window, greatly increasing the number of training samples.
        * Compute time differences relative to the query time (last event in
          the prefix) and store them for time-aware positional encoding.

        Example:
            User sequence: [item1, item2, item3, item4, item5]
            Timestamps: [t1, t2, t3, t4, t5]
            Generated samples:
                ([item1], item2, time_diffs=[0, t2 - t1])
                ([item1, item2], item3, time_diffs=[0, t2 - t1, t3 - t2])
                ([item1, item2, item3], item4, time_diffs=[0, t2 - t1, t3 - t2, t4 - t3])
                ([item1, item2, item3, item4], item5,
                 time_diffs=[0, t2 - t1, t3 - t2, t4 - t3, t5 - t4])

        Args:
            user_sequences (dict): User sequences {user_id: [movie_id1, ...]}.
            user_timestamps (dict): User timestamps {user_id: [timestamp1, ...]}.
            vocab (dict): Mapping from ``movie_id`` to ``token_id``.

        Returns:
            tuple: ``(seq_tokens, seq_positions, seq_time_diffs, targets, user_ids)``.
        """
        print("\n使用滑动窗口生成训练样本（支持时间戳）...")

        seq_tokens_list = []
        seq_positions_list = []
        seq_time_diffs_list = []  # 新增：时间差列表
        targets_list = []
        user_ids_list = []

        total_samples = 0
        for user_id, sequence in user_sequences.items():
            timestamps = user_timestamps[user_id]

            # 从每个用户序列生成多个样本
            # 对于长度为N的序列，生成N-1个样本
            for end_idx in range(1, len(sequence)):
                # 历史序列：从开始到end_idx（不包含）
                history = sequence[:end_idx]
                history_timestamps = timestamps[:end_idx]
                # 目标：end_idx位置的item
                target = sequence[end_idx]

                # 转换为token
                seq_tokens = [vocab.get(movie_id, 0) for movie_id in history]
                target_token = vocab.get(target, 0)

                # 计算时间差（秒）- 相对于查询时间（序列最后一个时间戳）
                # 参考Meta官方实现：query_time - timestamps
                query_timestamp = history_timestamps[-1]  # 序列最后一个时间戳作为查询时间
                seq_time_diffs = [query_timestamp - ts for ts in history_timestamps]
                # 例如：[100, 200, 300, 400] → [300, 200, 100, 0]

                # 截断到max_seq_len（保留最近的历史）
                if len(seq_tokens) > self.max_seq_len:
                    seq_tokens = seq_tokens[-self.max_seq_len:]
                    seq_time_diffs = seq_time_diffs[-self.max_seq_len:]

                # 填充到max_seq_len
                seq_len = len(seq_tokens)
                pad_len = self.max_seq_len - seq_len
                seq_tokens = [0] * pad_len + seq_tokens
                seq_time_diffs = [0] * pad_len + seq_time_diffs  # time difference is 0 at padding positions

                # 位置编码
                seq_positions = list(range(self.max_seq_len))

                seq_tokens_list.append(seq_tokens)
                seq_positions_list.append(seq_positions)
                seq_time_diffs_list.append(seq_time_diffs)
                targets_list.append(target_token)
                user_ids_list.append(user_id)
                total_samples += 1

        seq_tokens = np.array(seq_tokens_list, dtype=np.int32)
        seq_positions = np.array(seq_positions_list, dtype=np.int32)
        seq_time_diffs = np.array(seq_time_diffs_list, dtype=np.int32)  # time differences per position
        targets = np.array(targets_list, dtype=np.int32)
        user_ids = np.array(user_ids_list, dtype=np.int32)

        # 统计时间差信息
        non_zero_time_diffs = seq_time_diffs[seq_time_diffs > 0]
        if len(non_zero_time_diffs) > 0:
            print("时间差统计（秒）:")
            print(f"  平均: {np.mean(non_zero_time_diffs):.1f}")
            print(f"  中位数: {np.median(non_zero_time_diffs):.1f}")
            print(f"  最小: {np.min(non_zero_time_diffs)}")
            print(f"  最大: {np.max(non_zero_time_diffs)}")
            print(f"  平均（小时）: {np.mean(non_zero_time_diffs) / 3600:.1f}")
            print(f"  平均（天）: {np.mean(non_zero_time_diffs) / 86400:.1f}")

        print(f"生成样本数: {total_samples}")
        print(f"数据增强倍数: {total_samples / len(user_sequences):.1f}x")

        return seq_tokens, seq_positions, seq_time_diffs, targets, user_ids

    def split_data_by_user(self, seq_tokens, seq_positions, seq_time_diffs, targets, user_ids, train_ratio=0.7, val_ratio=0.1):
        """Split data into train/val/test sets at the user level.

        Important: we split on users instead of individual samples to avoid
        information leakage (the same user's interactions appearing in both
        train and test sets).

        Args:
            seq_tokens (ndarray): Sequence tokens.
            seq_positions (ndarray): Position indices.
            seq_time_diffs (ndarray): Time-difference sequences.
            targets (ndarray): Target tokens.
            user_ids (ndarray): User IDs per sample.
            train_ratio (float): Fraction of users assigned to the training set.
            val_ratio (float): Fraction of users assigned to the validation set.

        Returns:
            dict: A dictionary with ``train``, ``val`` and ``test`` splits.
        """
        print("\n按用户分割数据集...")

        # 获取唯一用户
        unique_users = np.unique(user_ids)
        n_users = len(unique_users)

        # 随机打乱用户
        np.random.shuffle(unique_users)

        # 分割用户
        train_size = int(n_users * train_ratio)
        val_size = int(n_users * val_ratio)

        train_users = set(unique_users[:train_size])
        val_users = set(unique_users[train_size:train_size + val_size])
        test_users = set(unique_users[train_size + val_size:])

        # 根据用户分割样本
        train_mask = np.array([uid in train_users for uid in user_ids])
        val_mask = np.array([uid in val_users for uid in user_ids])
        test_mask = np.array([uid in test_users for uid in user_ids])

        data_split = {
            'train': {
                'seq_tokens': seq_tokens[train_mask],
                'seq_positions': seq_positions[train_mask],
                'seq_time_diffs': seq_time_diffs[train_mask],  # time differences for each prefix
                'targets': targets[train_mask]
            },
            'val': {
                'seq_tokens': seq_tokens[val_mask],
                'seq_positions': seq_positions[val_mask],
                'seq_time_diffs': seq_time_diffs[val_mask],  # time differences for each prefix
                'targets': targets[val_mask]
            },
            'test': {
                'seq_tokens': seq_tokens[test_mask],
                'seq_positions': seq_positions[test_mask],
                'seq_time_diffs': seq_time_diffs[test_mask],  # time differences for each prefix
                'targets': targets[test_mask]
            }
        }

        print(f"用户分割: Train={len(train_users)}, Val={len(val_users)}, Test={len(test_users)}")
        print(f"样本分割: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

        return data_split

    def save_data(self, data_split, vocab):
        """Save processed data splits and vocabulary to disk.

        Args:
            data_split (dict): Dictionary containing train/val/test splits.
            vocab (dict): Mapping from ``movie_id`` to ``token_id``.
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

        # 5. 使用滑动窗口生成训练样本（包含时间差）
        seq_tokens, seq_positions, seq_time_diffs, targets, user_ids = self.generate_training_samples_sliding_window(user_sequences, user_timestamps, vocab)

        # 6. 按用户分割数据
        data_split = self.split_data_by_user(seq_tokens, seq_positions, seq_time_diffs, targets, user_ids)

        # 7. 保存数据
        self.save_data(data_split, vocab)

        print("=" * 80)
        print("预处理完成！")
        print("=" * 80)
        print("\n关键改进:")
        print("✅ 采用滑动窗口策略，大幅提升数据量")
        print("✅ 添加冷启动过滤，提高数据质量")
        print("✅ 按用户分割数据，避免数据泄露")
        print("✅ 序列长度统计，便于调优")
        print("✅ 支持时间戳处理，计算时间差用于时间感知建模")  # 新增


if __name__ == '__main__':
    # Use the current directory as the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir
    output_dir = os.path.join(script_dir, 'processed')

    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    preprocessor = MovieLensHSTUPreprocessor(data_dir=data_dir, output_dir=output_dir)
    preprocessor.preprocess()
