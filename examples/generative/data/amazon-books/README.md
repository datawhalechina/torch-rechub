# Amazon Books Dataset for HLLM

This directory contains data preprocessing scripts for the Amazon Books dataset, following the [ByteDance HLLM official implementation](https://github.com/bytedance/HLLM).

## Dataset Information

The Amazon Books dataset is one of the official datasets used in the HLLM paper. It contains book reviews and metadata from Amazon.

### Data Sources

1. **Interactions (ratings_Books.csv)**:
   - Raw data: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv
   - Processed by ByteDance: https://huggingface.co/ByteDance/HLLM/resolve/main/Interactions/amazon_books.csv

2. **Item Information (meta_Books.json.gz)**:
   - Raw data: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
   - Processed by ByteDance: https://huggingface.co/ByteDance/HLLM/resolve/main/ItemInformation/amazon_books.csv

### Data Format

**ratings_Books.csv** (CSV format):
```
user_id,item_id,rating,timestamp
```

**meta_Books.json.gz** (JSON Lines format):
```json
{"asin": "...", "title": "...", "description": "..."}
```

## Quick Start

### Step 1: Download and Preprocess HSTU Format Data

The preprocessing scripts can download data automatically. By default they:

- use ByteDance processed data (`--data_source bytedance`)
- download the required file
- skip download when the target file already exists

Use `--data_source raw` to use Stanford SNAP raw data instead. Use `--no_download` to skip download and process an existing local file. Use `--overwrite` to refresh an existing downloaded file.

```bash
# Default: ByteDance processed interactions, download only if missing.
# Interaction-count filtering is skipped for ByteDance data by default.
python preprocess_amazon_books.py \
    --data_source bytedance \
    --data_dir . \
    --output_dir ./processed

# Stanford SNAP raw interactions
# Interaction-count filtering is enabled for raw data by default.
python preprocess_amazon_books.py \
    --data_source raw \
    --data_dir . \
    --output_dir ./processed

# Reuse an existing local file without downloading
python preprocess_amazon_books.py \
    --data_source bytedance \
    --no_download \
    --data_dir . \
    --output_dir ./processed

# Force re-download
python preprocess_amazon_books.py \
    --data_source bytedance \
    --overwrite \
    --data_dir . \
    --output_dir ./processed

```

Output files:
- `processed/vocab.pkl` - Item vocabulary
- `processed/train_data.pkl` - Training sequences
- `processed/val_data.pkl` - Validation sequences
- `processed/test_data.pkl` - Test sequences

Expected local interaction files:
- `raw`: `ratings_Books.csv`
- `bytedance`: `amazon_books_interactions.csv`
  - If manually downloaded, `amazon_books.csv` is also accepted as a fallback.
  - Supported interaction columns:
    - `user_id,item_id,rating,timestamp`
    - `item_id,user_id,timestamp` (ByteDance processed format)
  - `bytedance` keeps the provided interactions unchanged, while `raw` applies users/items interaction-count filtering.

### Step 2: Generate HLLM Item Embeddings

```bash
# Default: ByteDance processed item information, download only if missing
python preprocess_amazon_books_hllm.py \
    --data_source bytedance \
    --model_type tinyllama \
    --device cuda

# Stanford SNAP raw item metadata
python preprocess_amazon_books_hllm.py \
    --data_source raw \
    --model_type tinyllama \
    --device cuda

# Reuse an existing local file without downloading
python preprocess_amazon_books_hllm.py \
    --data_source bytedance \
    --no_download \
    --model_type tinyllama \
    --device cuda

# Force re-download
python preprocess_amazon_books_hllm.py \
    --data_source bytedance \
    --overwrite \
    --model_type tinyllama \
    --device cuda
```

Output files:
- `processed/item_text_map.pkl` - Item text descriptions
- `processed/item_embeddings_tinyllama.pt` - Pre-computed item embeddings

Expected local item information files:
- `raw`: `meta_Books.json.gz` or `meta_Books.json`
- `bytedance`: `item_information.csv`
  - If manually downloaded, `amazon_books_items.csv` or `amazon_books.csv` is also accepted when it has columns `item_id,description,title`.

### Step 3: Train HLLM Model

```bash
cd ../..
python run_hllm_amazon_books.py --device cuda --epochs 10
```

## File Structure

```
amazon-books/
├── README.md
├── preprocess_amazon_books.py      # HSTU format preprocessing
├── preprocess_amazon_books_hllm.py # HLLM embeddings generation
├── ratings_Books.csv               # Interactions (downloaded raw or ByteDance)
├── meta_Books.json.gz              # Raw metadata (downloaded when --data_source raw)
├── item_information.csv            # ByteDance item information (downloaded when --data_source bytedance)
└── processed/                      # Preprocessed output
    ├── vocab.pkl
    ├── train_data.pkl
    ├── val_data.pkl
    ├── test_data.pkl
    ├── item_text_map.pkl
    └── item_embeddings_tinyllama.pt
```

## Notes

- The official HLLM implementation filters users and items with >= 5 interactions
- Text format: `"Compress the following sentence into embedding: title: {title}description: {description}"` (no 'tag' field for books)
- This implementation is compatible with the official ByteDance HLLM data format
