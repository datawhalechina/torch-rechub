"""MovieLens-1M数据预处理脚本 - 用于HSTU模型.

该脚本将MovieLens-1M原始数据转换为HSTU模型所需的格式：
- 按用户构建行为序列
- 按时间排序
- 使用滑动窗口生成训练样本（大幅提升数据量）
- 生成train/val/test分割
- 输出seq_tokens, seq_positions, targets

参考实现: https://github.com/meta-recsys/generative-recommenders
关键改进: 采用滑动窗口策略，从每个用户序列生成多个训练样本
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict


class MovieLensHSTUPreprocessor:
    """MovieLens-1M数据预处理器 - HSTU格式.

    采用滑动窗口策略，从每个用户序列生成多个训练样本，
    大幅提升数据利用率和模型训练效果。

    示例：
        用户序列: [item1, item2, item3, item4, item5]
        生成样本:
            ([item1], item2)
            ([item1, item2], item3)
            ([item1, item2, item3], item4)
            ([item1, item2, item3, item4], item5)
    """

    def __init__(self, data_dir="./data/ml-1m/", output_dir="./data/ml-1m/processed/"):
        """初始化预处理器.

        Args:
            data_dir (str): 原始数据目录
            output_dir (str): 输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 参数
        self.min_seq_len = 5  # 最小序列长度（用户至少要有5个交互）
        self.max_seq_len = 200  # 最大序列长度（减少到200以提高效率）
        self.rating_threshold = 3  # 评分阈值（>=3为正样本）
        self.min_item_count = 5  # item至少出现5次（冷启动过滤）

    def load_data(self):
        """加载MovieLens-1M数据."""
        print("加载MovieLens-1M数据...")

        # 读取ratings数据
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(
            os.path.join(self.data_dir, 'ratings.dat'),
            sep='::',
            header=None,
            names=rnames,
            engine='python',
            encoding='ISO-8859-1'
        )

        # 读取movies数据
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(
            os.path.join(self.data_dir, 'movies.dat'),
            sep='::',
            header=None,
            names=mnames,
            engine='python',
            encoding='ISO-8859-1'
        )

        print(f"原始数据: {len(ratings)} 条评分, {ratings['user_id'].nunique()} 用户, {ratings['movie_id'].nunique()} 电影")

        return ratings, movies

    def filter_data(self, ratings):
        """过滤数据：评分阈值 + 冷启动过滤.

        Args:
            ratings (pd.DataFrame): 原始评分数据

        Returns:
            pd.DataFrame: 过滤后的评分数据
        """
        print("\n数据过滤...")
        print(f"过滤前: {len(ratings)} 条评分")

        # 1. 过滤低评分（只保留>=3的评分）
        ratings = ratings[ratings['rating'] >= self.rating_threshold]
        print(f"评分过滤后 (>={self.rating_threshold}): {len(ratings)} 条评分")

        # 2. 冷启动过滤：item至少出现min_item_count次
        item_counts = ratings['movie_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_count].index
        ratings = ratings[ratings['movie_id'].isin(valid_items)]
        print(f"Item过滤后 (>={self.min_item_count}次): {len(ratings)} 条评分")

        # 3. 冷启动过滤：user至少有min_seq_len个交互
        user_counts = ratings['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_seq_len].index
        ratings = ratings[ratings['user_id'].isin(valid_users)]
        print(f"User过滤后 (>={self.min_seq_len}次): {len(ratings)} 条评分")

        print(f"过滤后: {ratings['user_id'].nunique()} 用户, {ratings['movie_id'].nunique()} 电影")

        return ratings

    def create_vocab(self, ratings):
        """创建电影ID到token的映射.

        Args:
            ratings (pd.DataFrame): 评分数据

        Returns:
            dict: movie_id -> token_id 的映射
        """
        print("\n创建词表...")

        unique_movies = sorted(ratings['movie_id'].unique())
        # token_id从1开始，0保留给PAD
        vocab = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movies)}
        vocab[0] = 0  # PAD token

        print(f"词表大小: {len(vocab)} (包含PAD)")
        return vocab

    def build_user_sequences(self, ratings):
        """构建用户序列（按时间排序）.

        Args:
            ratings (pd.DataFrame): 评分数据

        Returns:
            tuple: (user_sequences, user_timestamps)
                - user_sequences: {user_id: [movie_id1, movie_id2, ...]}
                - user_timestamps: {user_id: [timestamp1, timestamp2, ...]}
        """
        print("\n构建用户序列...")

        # 按时间排序
        ratings_sorted = ratings.sort_values(['user_id', 'timestamp'])

        # 按用户分组，同时保存movie_id和timestamp
        user_sequences = {}
        user_timestamps = {}
        for user_id, group in ratings_sorted.groupby('user_id'):
            user_sequences[user_id] = group['movie_id'].tolist()
            user_timestamps[user_id] = group['timestamp'].tolist()

        # 统计序列长度
        seq_lengths = [len(seq) for seq in user_sequences.values()]
        print(f"用户数: {len(user_sequences)}")
        print(f"序列长度统计: 平均={np.mean(seq_lengths):.1f}, 最小={np.min(seq_lengths)}, 最大={np.max(seq_lengths)}")

        return user_sequences, user_timestamps

    def generate_training_samples_sliding_window(self, user_sequences, user_timestamps, vocab):
        """使用滑动窗口生成训练样本（支持时间戳）.

        关键改进：从每个用户序列生成多个训练样本，大幅提升数据量。
        新增功能：计算时间差并保存，用于时间感知的位置编码。

        示例：
            用户序列: [item1, item2, item3, item4, item5]
            时间戳: [t1, t2, t3, t4, t5]
            生成样本:
                ([item1], item2, time_diffs=[0, t2-t1])
                ([item1, item2], item3, time_diffs=[0, t2-t1, t3-t2])
                ([item1, item2, item3], item4, time_diffs=[0, t2-t1, t3-t2, t4-t3])
                ([item1, item2, item3, item4], item5, time_diffs=[0, t2-t1, t3-t2, t4-t3, t5-t4])

        Args:
            user_sequences (dict): 用户序列字典 {user_id: [movie_id1, ...]}
            user_timestamps (dict): 用户时间戳字典 {user_id: [timestamp1, ...]}
            vocab (dict): 词表映射

        Returns:
            tuple: (seq_tokens, seq_positions, seq_time_diffs, targets, user_ids)
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
                seq_time_diffs = [0] * pad_len + seq_time_diffs  # padding位置的时间差为0

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
        seq_time_diffs = np.array(seq_time_diffs_list, dtype=np.int32)  # 新增
        targets = np.array(targets_list, dtype=np.int32)
        user_ids = np.array(user_ids_list, dtype=np.int32)

        # 统计时间差信息
        non_zero_time_diffs = seq_time_diffs[seq_time_diffs > 0]
        if len(non_zero_time_diffs) > 0:
            print(f"时间差统计（秒）:")
            print(f"  平均: {np.mean(non_zero_time_diffs):.1f}")
            print(f"  中位数: {np.median(non_zero_time_diffs):.1f}")
            print(f"  最小: {np.min(non_zero_time_diffs)}")
            print(f"  最大: {np.max(non_zero_time_diffs)}")
            print(f"  平均（小时）: {np.mean(non_zero_time_diffs) / 3600:.1f}")
            print(f"  平均（天）: {np.mean(non_zero_time_diffs) / 86400:.1f}")

        print(f"生成样本数: {total_samples}")
        print(f"数据增强倍数: {total_samples / len(user_sequences):.1f}x")

        return seq_tokens, seq_positions, seq_time_diffs, targets, user_ids

    def split_data_by_user(self, seq_tokens, seq_positions, seq_time_diffs, targets, user_ids,
                           train_ratio=0.7, val_ratio=0.1):
        """按用户分割数据集（避免数据泄露）.

        重要：按用户分割而不是按样本随机分割，避免同一用户的数据出现在训练集和测试集中。

        Args:
            seq_tokens (ndarray): 序列tokens
            seq_positions (ndarray): 序列位置
            seq_time_diffs (ndarray): 序列时间差
            targets (ndarray): 目标tokens
            user_ids (ndarray): 用户IDs
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例

        Returns:
            dict: 包含train/val/test数据的字典
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
                'seq_time_diffs': seq_time_diffs[train_mask],  # 新增
                'targets': targets[train_mask]
            },
            'val': {
                'seq_tokens': seq_tokens[val_mask],
                'seq_positions': seq_positions[val_mask],
                'seq_time_diffs': seq_time_diffs[val_mask],  # 新增
                'targets': targets[val_mask]
            },
            'test': {
                'seq_tokens': seq_tokens[test_mask],
                'seq_positions': seq_positions[test_mask],
                'seq_time_diffs': seq_time_diffs[test_mask],  # 新增
                'targets': targets[test_mask]
            }
        }

        print(f"用户分割: Train={len(train_users)}, Val={len(val_users)}, Test={len(test_users)}")
        print(f"样本分割: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

        return data_split

    def save_data(self, data_split, vocab):
        """保存处理后的数据.

        Args:
            data_split (dict): 分割后的数据
            vocab (dict): 词表映射
        """
        print("\n保存数据...")

        # 保存数据
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
        """执行完整的预处理流程."""
        print("=" * 80)
        print("MovieLens-1M HSTU数据预处理 - 滑动窗口版本（支持时间戳）")
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
        seq_tokens, seq_positions, seq_time_diffs, targets, user_ids = self.generate_training_samples_sliding_window(
            user_sequences, user_timestamps, vocab
        )

        # 6. 按用户分割数据
        data_split = self.split_data_by_user(
            seq_tokens, seq_positions, seq_time_diffs, targets, user_ids
        )

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
    # 使用当前目录作为数据目录
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir
    output_dir = os.path.join(script_dir, 'processed')

    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    preprocessor = MovieLensHSTUPreprocessor(data_dir=data_dir, output_dir=output_dir)
    preprocessor.preprocess()

