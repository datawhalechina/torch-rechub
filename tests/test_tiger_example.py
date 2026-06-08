"""Tests for the TIGER examples: TigerSeqDataset / Trie token+collate logic and
a toy end-to-end smoke run of the MovieLens example script.

The dataset / Trie / collate tests are offline and always run. The example
script imports ``transformers`` at module load and the full smoke run needs the
``t5-small`` weights, so those are skipped when unavailable.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torch_rechub.utils.data import TigerSeqDataset, Trie

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "generative"

# Small, fully-specified toy dataset (5 items per user, 2 levels per semantic id).
INTERS = {"u0": [1, 2, 3, 4, 5], "u1": [2, 3, 4, 5, 6]}
INDICES = {
    "1": ["<a_0>",
          "<b_0>"],
    "2": ["<a_0>",
          "<b_1>"],
    "3": ["<a_1>",
          "<b_0>"],
    "4": ["<a_1>",
          "<b_1>"],
    "5": ["<a_2>",
          "<b_0>"],
    "6": ["<a_2>",
          "<b_1>"],
}


class StubTokenizer:
    """Minimal stand-in for a T5 tokenizer that maps each ``<x_y>`` semantic-id
    token to a unique positive integer. Only implements what ``get_collate_fn``
    needs, so the collate logic can be tested without downloading T5.
    """

    pad_token_id = 0
    model_max_length = 64

    def __init__(self):
        self._vocab = {}
        self._next = 1

    def _ids(self, text):
        ids = []
        for tok in re.findall(r"<[^>]*>", text):
            if tok not in self._vocab:
                self._vocab[tok] = self._next
                self._next += 1
            ids.append(self._vocab[tok])
        return ids

    def __call__(self, texts, return_tensors=None, padding=None, max_length=None, truncation=None, return_attention_mask=None):
        seqs = [self._ids(t) for t in texts]
        max_len = max((len(s) for s in seqs), default=0)
        if max_length is not None and truncation:
            max_len = min(max_len, max_length)
        input_ids, attention = [], []
        for s in seqs:
            s = s[:max_len]
            pad = max_len - len(s)
            input_ids.append(s + [self.pad_token_id] * pad)
            attention.append([1] * len(s) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention)}


# =========================================================
# TigerSeqDataset
# =========================================================
def test_train_mode_expands_history_prefixes():
    data = TigerSeqDataset(INTERS, INDICES, max_his_len=20, mode="train")
    # items[:-2] keeps the first 3 items; range(1, 3) yields 2 samples per user.
    assert len(data) == 4
    sample = data[0]
    assert sample["input_ids"] == "<a_0><b_0>"  # u0 history = item 1
    assert sample["labels"] == "<a_0><b_1>"  # next item = item 2


def test_valid_and_test_modes_use_holdout_targets():
    valid = TigerSeqDataset(INTERS, INDICES, max_his_len=20, mode="valid")
    test = TigerSeqDataset(INTERS, INDICES, max_his_len=20, mode="test")
    assert len(valid) == 2 and len(test) == 2

    # valid target is the second-to-last item, test target is the last item.
    assert valid[0]["labels"] == "<a_1><b_1>"  # u0 item 4
    assert test[0]["labels"] == "<a_2><b_0>"  # u0 item 5
    # test history covers the first 4 items joined together.
    assert test[0]["input_ids"] == "<a_0><b_0><a_0><b_1><a_1><b_0><a_1><b_1>"


def test_max_his_len_truncates_history():
    data = TigerSeqDataset(INTERS, INDICES, max_his_len=1, mode="test")
    # Only the most recent history item is kept.
    assert data[0]["input_ids"] == "<a_1><b_1>"  # u0 item 4 (last before holdout)


def test_get_new_tokens_is_sorted_unique():
    data = TigerSeqDataset(INTERS, INDICES, max_his_len=20, mode="train")
    assert data.get_new_tokens() == ["<a_0>", "<a_1>", "<a_2>", "<b_0>", "<b_1>"]


def test_get_all_items_returns_joined_semantic_ids():
    data = TigerSeqDataset(INTERS, INDICES, max_his_len=20, mode="train")
    items = data.get_all_items()
    assert "<a_0><b_0>" in items
    assert len(items) == 6  # one per distinct semantic id


# =========================================================
# Collate
# =========================================================
def test_collate_masks_pad_label_positions():
    data = TigerSeqDataset(INTERS, INDICES, max_his_len=20, mode="train")
    collate = data.get_collate_fn(StubTokenizer())

    batch = [
        {"input_ids": "<a_0><b_0>", "labels": "<a_0>"},  # 1 label token
        {"input_ids": "<a_0><b_0><a_1>", "labels": "<a_0><b_1><c_2>"},  # 3 label tokens
    ]
    out = collate(batch)

    # Inputs are padded to the longest sequence (3 tokens).
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"][0].tolist() == [1, 1, 0]

    # Labels are padded to 3; the short row's 2 pad slots become -100.
    assert out["labels"].shape == (2, 3)
    assert out["labels"][0].tolist()[1:] == [-100, -100]
    assert (out["labels"] == 0).sum().item() == 0  # no leftover pad id 0 in labels


# =========================================================
# Trie
# =========================================================
def test_trie_constrains_next_tokens():
    # Two candidate item id sequences sharing a prefix token.
    trie = Trie([[0, 10, 20], [0, 10, 21], [0, 11, 30]])
    # After [0, 10] the only legal next tokens are 20 and 21.
    assert sorted(trie.get([0, 10])) == [20, 21]
    # After [0] the legal first tokens are 10 and 11.
    assert sorted(trie.get([0])) == [10, 11]
    assert len(trie) == 3


# =========================================================
# Example script: toy data generation
# =========================================================
def _load_example(name):
    path = EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_movielens_toy_data_builds_valid_dataset(tmp_path):
    module = _load_example("run_tiger_movielens.py")
    inter_path = tmp_path / "inter.json"
    indice_path = tmp_path / "semantic_ids.json"
    args = SimpleNamespace(
        data_inter_path=str(inter_path),
        data_indice_path=str(indice_path),
        toy_num_users=12,
        toy_num_items=16,
        toy_seed=0,
    )
    module.generate_toy_data(args)

    assert inter_path.exists() and indice_path.exists()

    import json

    inters = json.loads(inter_path.read_text())
    indices = json.loads(indice_path.read_text())
    assert len(inters) == 12
    # Every item referenced in inter.json must have a semantic id.
    referenced = {i for seq in inters.values() for i in seq}
    assert all(str(i) in indices for i in referenced)

    data = TigerSeqDataset(inters, indices, max_his_len=20, mode="train")
    assert len(data) > 0
    assert all(tok.startswith("<") for tok in data.get_new_tokens())


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_movielens_prepare_data_from_ratings(tmp_path):
    module = _load_example("run_tiger_movielens.py")
    ratings = tmp_path / "ratings.dat"
    # UserID::MovieID::Rating::Timestamp — u1 has 3 interactions, u2 has 1.
    ratings.write_text("1::101::5::3\n1::103::4::1\n1::102::3::2\n2::101::5::9\n")
    inter_path = tmp_path / "inter.json"
    args = SimpleNamespace(
        ratings_path=str(ratings),
        data_inter_path=str(inter_path),
        data_indice_path=str(tmp_path / "semantic_ids.json"),
        min_seq_len=3,
        max_his_len=20,
    )
    module.prepare_data(args)

    import json

    inters = json.loads(inter_path.read_text())
    # Only u1 has >= 3 interactions; items ordered by timestamp (103,102,101).
    assert list(inters.keys()) == ["1"]
    # movie 103 (ts=1) -> item id by sorted movie id: 101->1, 102->2, 103->3.
    assert inters["1"] == [3, 2, 1]


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_movielens_prepare_data_aligns_to_vocab(tmp_path):
    """With --vocab_path, inter.json item ids are the vocab token ids (not a
    standalone sorted mapping), and movies absent from the vocab are skipped."""
    module = _load_example("run_tiger_movielens.py")
    import json
    import pickle

    # Non-contiguous movie->token mapping proves we use the vocab, not sorting.
    vocab = {"item_to_idx": {10: 1, 20: 2, 30: 3}, "idx_to_item": {1: 10, 2: 20, 3: 30}}
    vocab_path = tmp_path / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    ratings = tmp_path / "ratings.dat"
    # u1: movies 30,10,20 (ts 3,1,2) -> chronological 10,20,30 -> token ids 1,2,3.
    # movie 99 has no token -> dropped.
    ratings.write_text("1::30::5::3\n1::10::4::1\n1::20::3::2\n1::99::5::4\n")
    inter_path = tmp_path / "inter.json"
    args = SimpleNamespace(
        ratings_path=str(ratings),
        data_inter_path=str(inter_path),
        data_indice_path=str(tmp_path / "semantic_ids.json"),
        vocab_path=str(vocab_path),
        min_seq_len=3,
        max_his_len=20,
    )
    module.prepare_data(args)

    inters = json.loads(inter_path.read_text())
    assert inters["1"] == [1, 2, 3]  # vocab token ids, chronological, movie 99 skipped


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_movielens_toy_end_to_end(tmp_path, monkeypatch):
    """Full generate -> train -> test smoke run on toy data. Skips when the
    ``t5-small`` weights cannot be loaded (e.g. offline CI)."""
    module = _load_example("run_tiger_movielens.py")
    try:
        from transformers import T5Tokenizer

        T5Tokenizer.from_pretrained("t5-small")
    except Exception as exc:  # noqa: BLE001 - network / cache miss => skip
        pytest.skip(f"t5-small unavailable: {exc}")

    argv = [
        "run_tiger_movielens.py",
        "--mode",
        "all",
        "--data_inter_path",
        str(tmp_path / "inter.json"),
        "--data_indice_path",
        str(tmp_path / "semantic_ids.json"),
        "--output_dir",
        str(tmp_path / "ckpt"),
        "--toy_num_users",
        "12",
        "--toy_num_items",
        "16",
        "--epochs",
        "1",
        "--per_device_batch_size",
        "4",
        "--gradient_accumulation_steps",
        "1",
        "--num_beams",
        "4",
        "--test_batch_size",
        "2",
        "--num_workers",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = module.parse_args()
    module.main(args)  # should run without raising
