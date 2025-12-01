"""Unified HLLM data preprocessing script for Amazon Books dataset.

This script combines item text extraction and item embedding generation into a single pipeline:
1. Extract product text information from Amazon Books metadata (meta_Books.json.gz)
2. Generate item embeddings using TinyLlama or Baichuan2
3. Save all necessary output files

Data format follows ByteDance HLLM official implementation:
- meta_Books.json.gz: {"asin": "...", "title": "...", "description": "..."}

Usage:
    python preprocess_amazon_books_hllm.py --model_type tinyllama --device cuda
    python preprocess_amazon_books_hllm.py --model_type baichuan2 --device cuda
"""

import gzip
import json
import os
import pickle

import numpy as np
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = _SCRIPT_DIR
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "processed")


def load_metadata(data_dir):
    """Load product metadata from meta_Books.json.gz or meta_Books.json."""
    # Try gzipped file first
    meta_file_gz = os.path.join(data_dir, "meta_Books.json.gz")
    meta_file = os.path.join(data_dir, "meta_Books.json")

    if os.path.exists(meta_file_gz):
        print(f"\n📖 Loading metadata from {meta_file_gz}...")
        open_func = gzip.open
        file_path = meta_file_gz
    elif os.path.exists(meta_file):
        print(f"\n📖 Loading metadata from {meta_file}...")
        open_func = open
        file_path = meta_file
    else:
        print("❌ Error: Metadata file not found")
        print("\nPlease download from:")
        print("  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz")
        print("Or use the processed version from ByteDance:")
        print("  https://huggingface.co/ByteDance/HLLM/resolve/main/ItemInformation/amazon_books.csv")
        return None

    metadata = {}
    with open_func(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc="Loading metadata"):
            try:
                # Handle both JSON and eval-style formats
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


def generate_embeddings(item_text_map, model_type, device, output_dir):
    """Generate item embeddings using LLM."""
    model_configs = {'tinyllama': {'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'embedding_dim': 2048}, 'baichuan2': {'model_name': 'baichuan-inc/Baichuan2-7B-Chat', 'embedding_dim': 4096}}

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    config = model_configs[model_type]
    model_name = config['model_name']

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == 'cuda' else torch.float32, device_map=device, trust_remote_code=True)
    model.eval()

    # Generate embeddings using official ByteDance HLLM approach
    # Uses last token's hidden state (no special [ITEM] token needed)
    # In official implementation, learnable embedding tokens are appended during training
    embeddings_list = []
    product_ids = list(item_text_map.keys())

    print(f"Generating embeddings for {len(product_ids)} products...")
    print("Using official ByteDance HLLM format (last token hidden state)")

    with torch.no_grad():
        for product_id in tqdm.tqdm(product_ids, desc="Generating embeddings"):
            text = item_text_map[product_id]
            # Text already contains ITEM_PROMPT prefix from extract_item_text()
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            # Use last token's hidden state as item embedding
            # This matches official implementation where item_emb_token_n=1
            # and embedding is extracted from the last position
            embedding = hidden_states[0, -1, :].cpu().numpy()
            embeddings_list.append(embedding)

    # Convert to tensor
    embeddings = torch.from_numpy(np.array(embeddings_list)).float()

    # Save embeddings
    output_file = os.path.join(output_dir, f"item_embeddings_{model_type}.pt")
    torch.save(embeddings, output_file)
    print(f"✅ Saved embeddings to {output_file}")
    print(f"   Shape: {embeddings.shape}")

    return embeddings


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unified HLLM preprocessing for Amazon Books")
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model_type", default="tinyllama", choices=["tinyllama", "baichuan2"], help="LLM model type")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")

    args = parser.parse_args()

    print("=" * 80)
    print("Amazon Books HLLM Preprocessing")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Extract item text
    metadata = load_metadata(args.data_dir)
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


if __name__ == "__main__":
    main()
