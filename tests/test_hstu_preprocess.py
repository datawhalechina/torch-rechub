"""MovieLens sampling, holdout, and next-token alignment regressions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_rechub.trainers.seq_trainer import SeqTrainer

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/generative/data/ml-1m/preprocess_ml_hstu.py'
spec = importlib.util.spec_from_file_location('preprocess_ml_hstu', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def prepare(tmp_path, mode, max_len=6):
    preprocessor = module.MovieLensHSTUPreprocessor(output_dir=str(tmp_path), training_sampling=mode)
    preprocessor.max_seq_len = max_len
    # Include a long history, a minimal eligible history, and a skipped user.
    sequences = {1: list(range(1, 9)), 2: [11, 12, 13, 14], 3: [21, 22, 23]}
    timestamps = {user: [100 * i * i for i in range(len(seq))] for user, seq in sequences.items()}
    vocab = {'item_to_idx': {i: i for i in range(1, 24)}}
    return preprocessor, preprocessor.generate_leave_last_out_samples(sequences, timestamps, vocab), vocab


def test_user_sequence_holdouts_and_supervision(tmp_path):
    _, splits, _ = prepare(tmp_path, 'user_sequence')
    train = splits['train']
    np.testing.assert_array_equal(train['seq_tokens'], [[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 11]])
    np.testing.assert_array_equal(train['targets'], [6, 12])
    np.testing.assert_array_equal(splits['val']['targets'], [7, 13])
    np.testing.assert_array_equal(splits['test']['targets'], [8, 14])
    # Exercise the real trainer's label construction, including PAD masking.
    trainer = object.__new__(SeqTrainer)
    labels = trainer._next_tokens(torch.tensor(train['seq_tokens']), torch.tensor(train['targets']))
    np.testing.assert_array_equal(labels, [[0, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 12]])
    for row in range(2):
        observed = set(train['seq_tokens'][row]) | {train['targets'][row]}
        assert splits['val']['targets'][row] not in observed
        assert splits['test']['targets'][row] not in observed


@pytest.mark.parametrize('max_len', [1, 3, 5, 6])
def test_recent_window_matches_last_prefix_and_preserves_evaluation(tmp_path, max_len):
    _, recent, _ = prepare(tmp_path, 'user_sequence', max_len)
    _, prefixes, _ = prepare(tmp_path, 'all_prefixes', max_len)
    assert len(prefixes['train']['targets']) == 6
    for key in recent['train']:
        np.testing.assert_array_equal(recent['train'][key], prefixes['train'][key][[4, 5]])
    for split in ('val', 'test'):
        for key in recent[split]:
            np.testing.assert_array_equal(recent[split][key], prefixes[split][key])


def test_truncation_keeps_timestamp_alignment(tmp_path):
    _, splits, _ = prepare(tmp_path, 'user_sequence', max_len=3)
    np.testing.assert_array_equal(splits['train']['seq_tokens'][0], [3, 4, 5])
    np.testing.assert_array_equal(splits['train']['seq_time_diffs'][0], [1200, 700, 0])
    assert splits['train']['targets'][0] == 6


def test_sampling_metadata_round_trip(tmp_path):
    preprocessor, splits, vocab = prepare(tmp_path, 'user_sequence')
    preprocessor.save_data(splits, vocab)
    metadata = json.loads((tmp_path / 'preprocessing_config.json').read_text())
    assert metadata == {'training_sampling': 'user_sequence', 'max_seq_len': 6, 'validation_holdout': 1, 'test_holdout': 1}


def test_unknown_sampling_rejected(tmp_path):
    with pytest.raises(ValueError, match='training_sampling'):
        module.MovieLensHSTUPreprocessor(output_dir=str(tmp_path), training_sampling='unknown')


@pytest.mark.parametrize('mode, expected_samples', [(None, 1), ('user_sequence', 1), ('all_prefixes', 5)])
def test_preprocess_cli_local_data(tmp_path, mode, expected_samples):
    import pickle
    import subprocess
    import sys

    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'ratings.dat').write_text(''.join(f'1::{i}::5::{i * 100}\n' for i in range(1, 9)))
    (raw / 'movies.dat').write_text(''.join(f'{i}::Movie {i}::Drama\n' for i in range(1, 9)))
    (raw / 'users.dat').write_text('1::F::1::10::00000\n')
    output = tmp_path / 'processed'
    command = [sys.executable, str(SCRIPT), '--data_dir', str(raw), '--output_dir', str(output), '--no_download']
    if mode is not None:
        command.extend(['--training_sampling', mode])
    subprocess.run(command, check=True, capture_output=True, text=True)
    with (output / 'train_data.pkl').open('rb') as f:
        train = pickle.load(f)
    assert train['seq_tokens'].shape == (expected_samples, 200)
    assert train['targets'][-1] == 6
    assert json.loads((output / 'preprocessing_config.json').read_text())['training_sampling'] == (mode or 'user_sequence')


def test_constructor_defaults_to_user_sequence(tmp_path):
    preprocessor = module.MovieLensHSTUPreprocessor(output_dir=str(tmp_path))
    assert preprocessor.training_sampling == 'user_sequence'
