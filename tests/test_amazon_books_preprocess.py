import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "generative" / "data" / "amazon-books" / "preprocess_amazon_books.py"


def _load_preprocess_module():
    spec = importlib.util.spec_from_file_location("preprocess_amazon_books", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HAS_PANDAS = importlib.util.find_spec("pandas") is not None
pytestmark = pytest.mark.skipif(not HAS_PANDAS, reason="pandas is not installed")


def _write_bytedance_csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("item_id,user_id,timestamp\n")
        for item_id, user_id, ts in rows:
            f.write(f"{item_id},{user_id},{ts}\n")


def test_read_interactions_coerces_numeric_item_id_to_str(tmp_path):
    module = _load_preprocess_module()
    csv_path = tmp_path / "amazon_books_interactions.csv"
    _write_bytedance_csv(csv_path, [(101, 1, 1000.0), (102, 1, 1001.0), (101, 2, 1002.0)])

    ratings = module.read_interactions(str(csv_path), expected_source="bytedance")

    assert ratings["item_id"].dtype == object
    assert ratings["user_id"].dtype == object
    assert set(ratings["item_id"].unique()) == {"101", "102"}
    assert set(ratings["user_id"].unique()) == {"1", "2"}


def test_read_interactions_str_keys_match_str_metadata_lookup(tmp_path):
    module = _load_preprocess_module()
    csv_path = tmp_path / "amazon_books_interactions.csv"
    _write_bytedance_csv(csv_path, [(7, 10, 1.0), (8, 10, 2.0), (7, 11, 3.0), (8, 11, 4.0), (9, 12, 5.0)])

    ratings = module.read_interactions(str(csv_path), expected_source="bytedance")
    sequences, vocab = module.build_sequences(ratings, max_seq_len=10, min_seq_len=2)
    item_to_idx = vocab["item_to_idx"]
    metadata = {"7": "title-7", "8": "title-8", "9": "title-9"}

    assert all(isinstance(k, str) for k in item_to_idx if k != "<PAD>")
    for item_key in item_to_idx:
        if item_key == "<PAD>":
            continue
        assert metadata.get(item_key) is not None


def test_read_interactions_rejects_source_schema_mismatch(tmp_path):
    module = _load_preprocess_module()
    csv_path = tmp_path / "ratings_Books.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("user_id,item_id,rating,timestamp\n")
        f.write("u1,i1,5,1000\n")

    with pytest.raises(ValueError, match=r"--data_source=bytedance .* raw schema"):
        module.read_interactions(str(csv_path), expected_source="bytedance")
