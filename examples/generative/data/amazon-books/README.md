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

### Step 1: Download Data

```bash
# Download from Stanford SNAP
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz

# Or download processed version from ByteDance HuggingFace
# See links above
```

### Step 2: Preprocess HSTU Format Data

```bash
python preprocess_amazon_books.py --data_dir . --output_dir ./processed
```

Output files:
- `processed/vocab.pkl` - Item vocabulary
- `processed/train_data.pkl` - Training sequences
- `processed/val_data.pkl` - Validation sequences
- `processed/test_data.pkl` - Test sequences

### Step 3: Generate HLLM Item Embeddings

```bash
python preprocess_amazon_books_hllm.py --model_type tinyllama --device cuda
```

Output files:
- `processed/item_text_map.pkl` - Item text descriptions
- `processed/item_embeddings_tinyllama.pt` - Pre-computed item embeddings

### Step 4: Train HLLM Model

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
├── ratings_Books.csv               # Raw interactions (download)
├── meta_Books.json.gz              # Raw metadata (download)
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
- Text format: `"Title: {title}. Description: {description}"` (no 'tag' field for books)
- This implementation is compatible with the official ByteDance HLLM data format

