"""Unified HLLM data preprocessing script for Amazon Books dataset.

This script combines item text extraction and item embedding generation into a single pipeline:
1. Extract product text information from Amazon Books metadata
2. Generate item embeddings using TinyLlama or Baichuan2
3. Save all necessary output files

Supported data sources:
- raw: Stanford SNAP meta_Books.json.gz
- bytedance: ByteDance HLLM processed ItemInformation/amazon_books.csv

Usage:
    python preprocess_amazon_books_hllm.py --data_source bytedance --model_type tinyllama --device cuda
    python preprocess_amazon_books_hllm.py --data_source raw --model_type tinyllama --device cuda
    python preprocess_amazon_books_hllm.py --no_download --model_type tinyllama --device cuda
"""

import gzip
import json
import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
import tqdm

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = _SCRIPT_DIR
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "processed")

METADATA_SOURCES = {
    "raw": {
        "url": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz",
        "filename": "meta_Books.json.gz",
        "description": "Stanford SNAP raw metadata",
    },
    "bytedance": {
        "url": "https://huggingface.co/ByteDance/HLLM/resolve/main/ItemInformation/amazon_books.csv",
        "filename": "item_information.csv",
        "description": "ByteDance HLLM processed item information",
    },
}


def download_file(url, output_path, overwrite=False):
    """Download a file to ``output_path``."""
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


def prepare_metadata_file(data_dir, data_source, download=True, overwrite=False):
    """Download the selected metadata file when requested."""
    source = METADATA_SOURCES[data_source]
    output_path = os.path.join(data_dir, source["filename"])

    if not download:
        print("Download disabled; using existing metadata file if available.")
        return True

    print(f"Selected metadata source: {data_source} ({source['description']})")
    return download_file(source["url"], output_path, overwrite=overwrite)


def get_metadata_file(data_dir, data_source):
    """Return the metadata path and detected format for the selected source."""
    if data_source == "bytedance":
        primary_path = os.path.join(data_dir, METADATA_SOURCES[data_source]["filename"])
        if os.path.exists(primary_path):
            return primary_path, "bytedance_csv"

        # Manual ByteDance downloads often keep the original amazon_books.csv name.
        for name in ["item_information.csv", "amazon_books_items.csv", "amazon_books.csv"]:
            fallback_path = os.path.join(data_dir, name)
            if os.path.exists(fallback_path) and _csv_has_columns(fallback_path, ["item_id", "description", "title"]):
                return fallback_path, "bytedance_csv"

        return primary_path, "bytedance_csv"

    meta_file_gz = os.path.join(data_dir, "meta_Books.json.gz")
    meta_file = os.path.join(data_dir, "meta_Books.json")

    if os.path.exists(meta_file_gz):
        return meta_file_gz, "json_gz"
    if os.path.exists(meta_file):
        return meta_file, "json"
    return meta_file_gz, "json_gz"


def load_bytedance_metadata_csv(file_path):
    """Load ByteDance processed item information CSV."""
    print(f"\n📖 Loading ByteDance item information from {file_path}...")
    items = pd.read_csv(file_path)
    expected_columns = ["item_id", "description", "title"]
    if items.columns.tolist() != expected_columns:
        print(f"❌ Error: Unexpected columns in {file_path}: {items.columns.tolist()}")
        print(f"   Expected: {expected_columns}")
        return None

    metadata = {}
    for _, row in tqdm.tqdm(items.iterrows(), total=len(items), desc="Loading item information"):
        product_id = str(row["item_id"])
        title = "" if pd.isna(row["title"]) else str(row["title"])
        description = "" if pd.isna(row["description"]) else str(row["description"])
        metadata[product_id] = {
            "asin": product_id,
            "title": title,
            "description": description,
        }

    print(f"✅ Loaded metadata for {len(metadata)} products")
    return metadata


def load_raw_metadata_json(file_path, file_format):
    """Load SNAP raw metadata from JSON Lines or gzipped JSON Lines."""
    print(f"\n📖 Loading metadata from {file_path}...")
    open_func = gzip.open if file_format == "json_gz" else open

    metadata = {}
    with open_func(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc="Loading metadata"):
            try:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = eval(line)

                product_id = item.get('asin')
                if product_id:
                    metadata[product_id] = item
            except Exception:
                continue

    print(f"✅ Loaded metadata for {len(metadata)} products")
    return metadata


def load_metadata(data_dir, data_source):
    """Load product metadata for the selected source."""
    file_path, file_format = get_metadata_file(data_dir, data_source)

    if not os.path.exists(file_path):
        print("❌ Error: Metadata file not found")
        print(f"   Expected file: {file_path}")
        print("\nRun with download enabled, or place an existing file in data_dir:")
        print("  python preprocess_amazon_books_hllm.py --data_source bytedance")
        print("  python preprocess_amazon_books_hllm.py --data_source raw")
        print("  python preprocess_amazon_books_hllm.py --no_download")
        return None

    if file_format == "bytedance_csv":
        return load_bytedance_metadata_csv(file_path)
    return load_raw_metadata_json(file_path, file_format)


# Official ByteDance HLLM item prompt
ITEM_PROMPT = "Compress the following sentence into embedding: "


def extract_item_text(metadata):
    """Extract text information from product metadata.

    Following official ByteDance HLLM format:
    "{item_prompt}title: {title}description: {description}"

    Note: Official format uses "key: value" without period separator.
    Books dataset doesn't use 'tag' field (unlike PixelRec).
    """
    item_text_map = {}

    print("Extracting item text...")
    for product_id, item in tqdm.tqdm(metadata.items(), desc="Extracting text"):
        title = item.get('title', '')
        description = item.get('description', '')

        # Handle description as list
        if isinstance(description, list):
            description = ' '.join(description)

        # Official ByteDance HLLM format:
        # "{item_prompt}title: {title}description: {description}"
        text = f"{ITEM_PROMPT}title: {title}description: {description}"
        item_text_map[product_id] = text

    return item_text_map


def generate_embeddings(item_text_map, model_type, device, output_dir, vocab_path=None):
    """Generate item embeddings using LLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if vocab_path is None:
        vocab_path = os.path.join(output_dir, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.pkl not found at {vocab_path}. " "Run preprocess_amazon_books.py first.")

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    item_to_idx = vocab['item_to_idx']
    vocab_size = max(item_to_idx.values()) + 1

    model_configs = {'tinyllama': {'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'embedding_dim': 2048}, 'baichuan2': {'model_name': 'baichuan-inc/Baichuan2-7B-Chat', 'embedding_dim': 4096}}

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    config = model_configs[model_type]
    model_name = config['model_name']
    d_model = config['embedding_dim']

    print(f"\nLoading model: {model_name}")
    print(f"vocab_size: {vocab_size}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == 'cuda' else torch.float32, device_map=device, trust_remote_code=True)
    model.eval()

    # Generate embeddings using official ByteDance HLLM approach
    # Uses last token's hidden state (no special [ITEM] token needed)
    # In official implementation, learnable embedding tokens are appended during training
    embeddings_array = np.zeros((vocab_size, d_model), dtype=np.float32)
    total_items = sum(1 for token_id in item_to_idx.values() if token_id != 0)
    missing = 0
    generated = 0

    print(f"Generating embeddings for {total_items} products...")
    print("Using official ByteDance HLLM format (last token hidden state)")

    with torch.no_grad():
        for product_id, token_id in tqdm.tqdm(sorted(item_to_idx.items(), key=lambda item: item[1]), desc="Generating embeddings"):
            if token_id == 0:
                continue

            text = item_text_map.get(product_id)
            if text is None:
                missing += 1
                continue

            # Text already contains ITEM_PROMPT prefix from extract_item_text()
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            # Use last token's hidden state as item embedding
            # This matches official implementation where item_emb_token_n=1
            # and embedding is extracted from the last position
            embedding = hidden_states[0, -1, :].cpu().numpy()
            embeddings_array[token_id] = embedding
            generated += 1

    # Convert to tensor
    embeddings = torch.from_numpy(embeddings_array).float()

    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"item_embeddings_{model_type}.pt")
    torch.save(embeddings, output_file)
    if missing:
        print(f"⚠️  {missing}/{total_items} ASIN 缺 metadata，留零向量")
    print(f"✅ Saved embeddings to {output_file}")
    print(f"   Shape: {tuple(embeddings.shape)} (vocab_size, d_model)")
    print(f"   Coverage: {generated}/{total_items} items")

    return embeddings


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unified HLLM preprocessing for Amazon Books")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--data_source", default="bytedance", choices=["raw", "bytedance"], help="Item metadata source to use")
    parser.add_argument("--no_download", action="store_true", help="Skip download and use an existing metadata file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing downloaded file")
    parser.add_argument("--model_type", default="tinyllama", choices=["tinyllama", "baichuan2"], help="LLM model type")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")

    args = parser.parse_args()

    print("=" * 80)
    print("Amazon Books HLLM Preprocessing")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data source: {args.data_source}")
    print(f"Download: {not args.no_download}")
    print(f"Overwrite downloads: {args.overwrite}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 0: Download or reuse metadata.
    prepare_metadata_file(
        args.data_dir,
        args.data_source,
        download=not args.no_download,
        overwrite=args.overwrite,
    )

    # Step 1: Extract item text
    metadata = load_metadata(args.data_dir, args.data_source)
    if metadata is None:
        return

    item_text_map = extract_item_text(metadata)

    # Save text map
    text_map_file = os.path.join(args.output_dir, "item_text_map.pkl")
    with open(text_map_file, 'wb') as f:
        pickle.dump(item_text_map, f)
    print(f"✅ Saved item text map to {text_map_file}")

    # Step 2: Generate embeddings
    generate_embeddings(item_text_map, args.model_type, args.device, args.output_dir)

    print("\n" + "=" * 80)
    print("✅ HLLM Preprocessing complete!")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print("  - item_text_map.pkl")
    print(f"  - item_embeddings_{args.model_type}.pt")
    print("\nPipeline order:")
    print("  1. python preprocess_amazon_books.py")
    print("  2. python preprocess_amazon_books_hllm.py --model_type tinyllama --device cuda")


if __name__ == "__main__":
    main()
