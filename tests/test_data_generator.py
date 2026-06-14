import numpy as np

from torch_rechub.utils.data import DataGenerator, SequenceDataGenerator


def test_data_generator_split_seed_is_reproducible():
    x = {"dense": np.arange(30)}
    y = np.arange(30)

    first = DataGenerator(x, y).generate_dataloader(split_ratio=[0.6, 0.2], batch_size=4, seed=2024, num_workers=0)
    second = DataGenerator(x, y).generate_dataloader(split_ratio=[0.6, 0.2], batch_size=4, seed=2024, num_workers=0)

    for first_loader, second_loader in zip(first, second):
        assert first_loader.dataset.indices == second_loader.dataset.indices


def test_sequence_data_generator_split_seed_is_reproducible():
    seq_tokens = np.arange(60).reshape(10, 6)
    seq_positions = np.tile(np.arange(6), (10, 1))
    targets = np.arange(10)
    seq_time_diffs = np.ones((10, 6))

    first = SequenceDataGenerator(seq_tokens, seq_positions, targets, seq_time_diffs).generate_dataloader(
        batch_size=2,
        split_ratio=(0.6, 0.2, 0.2),
        seed=2024,
        num_workers=0,
    )
    second = SequenceDataGenerator(seq_tokens, seq_positions, targets, seq_time_diffs).generate_dataloader(
        batch_size=2,
        split_ratio=(0.6, 0.2, 0.2),
        seed=2024,
        num_workers=0,
    )

    for first_loader, second_loader in zip(first, second):
        assert first_loader.dataset.indices == second_loader.dataset.indices
