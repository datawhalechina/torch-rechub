# Amazon Reviews 2014 for RPG

Preprocessing for the four Amazon Reviews 2014 categories benchmarked in the
RPG paper ([Generating Long Semantic IDs in Parallel for Recommendation](https://arxiv.org/abs/2506.05781), KDD 2025).

## Dataset

The 5-core reviews and the product metadata come from Stanford SNAP:

- Interactions: `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{category}_5.json.gz`
- Item metadata: `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{category}.json.gz`

Every review counts as one interaction, and a user's interactions are ordered
by `unixReviewTime`. Splitting is leave-one-out and happens later, inside
`RPGSeqDataset`: the last item of each sequence is the test target, the
second-to-last is the validation target.

| Category | Users | Items | Interactions | Avg. length |
| --- | --- | --- | --- | --- |
| `Beauty` | 22,363 | 12,101 | 198,502 | 8.9 |
| `Sports_and_Outdoors` | 35,598 | 18,357 | 296,337 | 8.3 |
| `Toys_and_Games` | 19,412 | 11,924 | 167,597 | 8.6 |
| `CDs_and_Vinyl` | 75,258 | 64,443 | 1,097,592 | 14.6 |

## Quick start

```bash
python preprocess_amazon_2014.py --category Beauty --device cuda
```

The script downloads the raw files (skipping any that already exist), builds
the user sequences, renders each item's metadata into a sentence, and encodes
those sentences with `sentence-transformers/sentence-t5-base`.

```bash
# Reuse already-downloaded raw files
python preprocess_amazon_2014.py --category Beauty --no_download

# Redo every step
python preprocess_amazon_2014.py --category Beauty --overwrite

# A different sentence encoder
python preprocess_amazon_2014.py --category Beauty --sent_emb_model sentence-transformers/all-mpnet-base-v2
```

## Outputs

```
amazon-2014/
├── raw/
│   ├── reviews_Beauty_5.json.gz
│   └── meta_Beauty.json.gz
└── processed/Beauty/
    ├── inter.json            # {user_id: [item_id, ...]}, chronological
    ├── id_mapping.json       # {"id2user": [...], "id2item": [...]}, index 0 = [PAD]
    ├── item_sentences.json   # [sentence, ...] for item ids 1..n_items-1
    └── item_embeddings.npy   # (n_items - 1, 768) float32
```

Item ids start at `1`; `0` is the padding id used by the model.

## Item metadata sentences

Each item's sentence concatenates, when present, the fields `title`, `price`,
`brand`, `feature`, `categories` and `description`. HTML tags, entities and
non-ASCII characters are stripped. Items absent from the metadata file get an
empty sentence and are still tokenized, since the quantizer only needs a
vector.

## Note on the sentence encoder

The paper's headline numbers use OpenAI `text-embedding-3-large` (3072-d,
reduced to 512 by whitened PCA). This script defaults to
`sentence-t5-base` (768-d, reduced to 128) so the pipeline runs without an API
key. On Beauty that costs roughly 9% relative NDCG@10 (0.0423 vs the published
0.0464); see the `Results` section of
`examples/generative/run_rpg_amazon_2014.py`.

## Next step

```bash
cd ../..
python run_rpg_amazon_2014.py --category Beauty --device cuda
```
