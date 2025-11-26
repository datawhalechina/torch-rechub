# Amazon Beauty Dataset for HLLM

This directory contains preprocessing scripts for the Amazon Beauty dataset for HLLM (Hierarchical Large Language Model for Recommendation).

## Quick Start

For complete instructions on downloading, preprocessing, and training with the Amazon Beauty dataset, please refer to the official documentation:

- **中文文档**: `docs/zh/blog/hllm_reproduction.md` (Section 5.4)
- **English Documentation**: `docs/en/blog/hllm_reproduction.md` (Section 5.4)

## Data Download

Download the Amazon Beauty dataset from: http://jmcauley.ucsd.edu/data/amazon/

You need two files:
1. `reviews_Beauty_5.json.gz` - User reviews with ratings and timestamps
2. `meta_Beauty.json.gz` - Product metadata (title, description, category, etc.)

Extract them to this directory:
```bash
cd examples/generative/data/amazon-beauty
gunzip reviews_Beauty_5.json.gz
gunzip meta_Beauty.json.gz
```

## Preprocessing Scripts

This directory contains two preprocessing scripts:

1. **`preprocess_amazon_beauty.py`** - Generates HSTU format sequence data
2. **`preprocess_amazon_beauty_hllm.py`** - Generates HLLM data (text extraction + embedding generation)

For detailed usage instructions, see the documentation linked above.

## Training Script

The training script is located at: `examples/generative/run_hllm_amazon_beauty.py`

For detailed usage instructions and parameter explanations, see the documentation linked above.

## References

- Amazon Review Data: http://jmcauley.ucsd.edu/data/amazon/
- HLLM Paper: https://arxiv.org/abs/2409.12740
- Official HLLM Code: https://github.com/bytedance/HLLM

## License

The Amazon Beauty dataset is provided by Julian McAuley and is subject to the terms of use specified on the original website.

