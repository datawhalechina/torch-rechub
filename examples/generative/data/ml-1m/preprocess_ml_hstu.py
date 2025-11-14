"""MovieLens-1M数据预处理脚本 - 用于HSTU模型.

该脚本将MovieLens-1M原始数据转换为HSTU模型所需的格式：
- 按用户构建行为序列
- 按时间排序
- 生成train/val/test分割
- 输出seq_tokens, seq_positions, targets
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from pathlib import Path


class MovieLensHSTUPreprocessor:
    """MovieLens-1M数据预处理器 - HSTU格式."""
    
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
        self.min_seq_len = 5  # 最小序列长度
        self.max_seq_len = 256  # 最大序列长度
        self.rating_threshold = 3  # 评分阈值（>=3为正样本）
    
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
        
        # 合并数据
        data = pd.merge(ratings, movies, on='movie_id')
        
        # 按评分过滤（只保留正样本）
        data = data[data['rating'] >= self.rating_threshold]
        
        # 按时间排序
        data = data.sort_values('timestamp')
        
        print(f"加载完成: {len(data)} 条交互记录")
        return data
    
    def build_sequences(self, data):
        """构建用户序列.
        
        Args:
            data (DataFrame): 交互数据
            
        Returns:
            dict: 用户序列字典 {user_id: [(movie_id, timestamp), ...]}
        """
        print("构建用户序列...")
        
        user_sequences = defaultdict(list)
        for _, row in data.iterrows():
            user_sequences[row['user_id']].append({
                'movie_id': row['movie_id'],
                'timestamp': row['timestamp']
            })
        
        # 过滤序列长度
        valid_sequences = {}
        for user_id, seq in user_sequences.items():
            if len(seq) >= self.min_seq_len:
                valid_sequences[user_id] = seq
        
        print(f"有效用户数: {len(valid_sequences)}")
        return valid_sequences
    
    def create_vocab(self, data):
        """创建电影ID到token的映射.
        
        Args:
            data (DataFrame): 交互数据
            
        Returns:
            dict: movie_id -> token_id 的映射
        """
        print("创建词表...")
        
        unique_movies = sorted(data['movie_id'].unique())
        vocab = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movies)}
        vocab[0] = 0  # PAD token
        
        print(f"词表大小: {len(vocab)}")
        return vocab
    
    def generate_training_data(self, user_sequences, vocab):
        """生成训练数据.
        
        Args:
            user_sequences (dict): 用户序列
            vocab (dict): 词表映射
            
        Returns:
            tuple: (seq_tokens, seq_positions, targets)
        """
        print("生成训练数据...")
        
        seq_tokens_list = []
        seq_positions_list = []
        targets_list = []
        
        for user_id, sequence in user_sequences.items():
            # 序列长度至少为2（至少有一个历史和一个目标）
            if len(sequence) < 2:
                continue
            
            # 使用所有历史预测最后一个
            history = sequence[:-1]
            target = sequence[-1]
            
            # 转换为token
            seq_tokens = [vocab.get(item['movie_id'], 0) for item in history]
            
            # 截断或填充到max_seq_len
            if len(seq_tokens) > self.max_seq_len:
                seq_tokens = seq_tokens[-self.max_seq_len:]
            
            # 填充到max_seq_len
            seq_len = len(seq_tokens)
            seq_tokens = [0] * (self.max_seq_len - seq_len) + seq_tokens
            
            # 位置编码
            seq_positions = list(range(self.max_seq_len))
            
            # 目标token
            target_token = vocab.get(target['movie_id'], 0)
            
            seq_tokens_list.append(seq_tokens)
            seq_positions_list.append(seq_positions)
            targets_list.append(target_token)
        
        seq_tokens = np.array(seq_tokens_list, dtype=np.int32)
        seq_positions = np.array(seq_positions_list, dtype=np.int32)
        targets = np.array(targets_list, dtype=np.int32)
        
        print(f"生成数据: {len(seq_tokens)} 条样本")
        return seq_tokens, seq_positions, targets
    
    def split_data(self, seq_tokens, seq_positions, targets, 
                   train_ratio=0.7, val_ratio=0.1):
        """分割train/val/test数据集.
        
        Args:
            seq_tokens (ndarray): 序列tokens
            seq_positions (ndarray): 序列位置
            targets (ndarray): 目标tokens
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
            
        Returns:
            dict: 包含train/val/test数据的字典
        """
        print("分割数据集...")
        
        n_samples = len(seq_tokens)
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        data_split = {
            'train': {
                'seq_tokens': seq_tokens[train_idx],
                'seq_positions': seq_positions[train_idx],
                'targets': targets[train_idx]
            },
            'val': {
                'seq_tokens': seq_tokens[val_idx],
                'seq_positions': seq_positions[val_idx],
                'targets': targets[val_idx]
            },
            'test': {
                'seq_tokens': seq_tokens[test_idx],
                'seq_positions': seq_positions[test_idx],
                'targets': targets[test_idx]
            }
        }
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        return data_split
    
    def save_data(self, data_split, vocab):
        """保存处理后的数据.
        
        Args:
            data_split (dict): 分割后的数据
            vocab (dict): 词表映射
        """
        print("保存数据...")
        
        # 保存数据
        for split_name, split_data in data_split.items():
            output_file = os.path.join(self.output_dir, f'{split_name}_data.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"保存 {split_name} 数据到 {output_file}")
        
        # 保存词表
        vocab_file = os.path.join(self.output_dir, 'vocab.pkl')
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"保存词表到 {vocab_file}")
    
    def preprocess(self):
        """执行完整的预处理流程."""
        print("=" * 60)
        print("MovieLens-1M HSTU数据预处理")
        print("=" * 60)
        
        # 加载数据
        data = self.load_data()
        
        # 构建序列
        user_sequences = self.build_sequences(data)
        
        # 创建词表
        vocab = self.create_vocab(data)
        
        # 生成训练数据
        seq_tokens, seq_positions, targets = self.generate_training_data(
            user_sequences, vocab
        )
        
        # 分割数据
        data_split = self.split_data(seq_tokens, seq_positions, targets)
        
        # 保存数据
        self.save_data(data_split, vocab)
        
        print("=" * 60)
        print("预处理完成！")
        print("=" * 60)


if __name__ == '__main__':
    preprocessor = MovieLensHSTUPreprocessor()
    preprocessor.preprocess()

