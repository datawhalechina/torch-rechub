"""Amazon Reviews 2014 preprocessing for RPG.

Downloads the 5-core reviews and the product metadata of one category from
Stanford SNAP, turns the reviews into chronological per-user item sequences,
and encodes every item's metadata into a sentence embedding. The embeddings
are what :class:`~torch_rechub.utils.opq.OPQTokenizer` quantizes into semantic
IDs.

Item ids start at ``1`` because ``0`` is the padding id used by the model.

Outputs (under ``<data_dir>/processed/<category>/``)
----------------------------------------------------
``inter.json``
    ``{user_id: [item_id, ...]}`` sorted by review time.
``id_mapping.json``
    ``{"id2item": [...], "id2user": [...]}``; index ``0`` is ``[PAD]``.
``item_sentences.json``
    ``[sentence, ...]`` for item ids ``1..n_items-1``.
``item_embeddings.npy``
    ``(n_items - 1, dim)`` float32 sentence embeddings, aligned with the
    sentences above.

Examples
--------
::

    python preprocess_amazon_2014.py --category Beauty --device cuda
    python preprocess_amazon_2014.py --category Toys_and_Games --no_download
"""

import argparse
import ast
import gzip
import html
import json
import os
import re
import urllib.request
from collections import defaultdict

import numpy as np
import tqdm

SNAP_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
# Categories benchmarked in the RPG paper. Any 2014 category on SNAP works.
RPG_CATEGORIES = ("Beauty", "Sports_and_Outdoors", "Toys_and_Games", "CDs_and_Vinyl")
META_FEATURES = ("title", "price", "brand", "feature", "categories", "description")


# =========================================================
# Download
# =========================================================
def download_file(url, output_path, overwrite=False):
    """Download a file to ``output_path`` with a progress bar."""
    if os.path.exists(output_path) and not overwrite:
        print(f"File exists, skip download: {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    print(f"Downloading: {url}")
    progress_bar = tqdm.tqdm(unit="B", unit_scale=True, desc=os.path.basename(output_path))

    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            progress_bar.total = total_size
        progress_bar.update(block_size)

    urllib.request.urlretrieve(url, tmp_path, reporthook=_hook)  # noqa: S310 - fixed https host
    progress_bar.close()
    os.replace(tmp_path, output_path)
    print(f"Downloaded: {output_path}")


# =========================================================
# Interactions
# =========================================================
def build_sequences(reviews_path):
    """Read the 5-core reviews into chronological per-user item-id sequences.

    Returns
    -------
    inter : dict
        ``{user_id: [item_id, ...]}`` with ids starting at 1.
    id_mapping : dict
        ``id2item`` / ``id2user`` lists whose index ``0`` holds ``[PAD]``.
    """
    print(f"Reading reviews: {reviews_path}")
    reviews = defaultdict(list)
    with gzip.open(reviews_path, "rt") as f:
        for line in tqdm.tqdm(f, desc="reviews"):
            record = json.loads(line)
            reviews[record["reviewerID"]].append((int(record["unixReviewTime"]), record["asin"]))

    id2user, id2item = ["[PAD]"], ["[PAD]"]
    item2id = {}
    inter = {}
    for user, events in reviews.items():
        events.sort()
        id2user.append(user)
        sequence = []
        for _, asin in events:
            if asin not in item2id:
                item2id[asin] = len(id2item)
                id2item.append(asin)
            sequence.append(item2id[asin])
        inter[len(id2user) - 1] = sequence

    print(f"users={len(inter)} items={len(id2item) - 1} interactions={sum(len(s) for s in inter.values())}")
    return inter, {"id2user": id2user, "id2item": id2item}


# =========================================================
# Item metadata sentences
# =========================================================
def clean_text(raw):
    """Strip HTML, control characters and non-ASCII noise from metadata text."""
    text = html.unescape(str(raw)).strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\x00-\x7F]|[\n\t]", " ", text)
    return re.sub(r" +", " ", text)


def feature_to_sentence(value):
    """Render one metadata field as a sentence fragment.

    Amazon metadata mixes types: ``price`` is a number, ``categories`` is a
    list of category paths, ``feature`` is a list of bullet points, and the
    rest are plain strings.
    """
    if isinstance(value, (int, float)):
        return f"{value}. "
    if isinstance(value, list) and value and isinstance(value[0], list):
        return ", ".join(clean_text(name) for path in value for name in path) + ". "
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value) + " "
    return clean_text(value) + " "


def build_sentences(meta_path, id2item):
    """Build one metadata sentence per item id, ordered by id.

    Items missing from the metadata file get an empty sentence.
    """
    print(f"Reading metadata: {meta_path}")
    wanted = set(id2item[1:])
    asin2sentence = {}
    with gzip.open(meta_path, "rt") as f:
        for line in tqdm.tqdm(f, desc="metadata"):
            # SNAP metadata lines are Python dict literals, not JSON.
            record = ast.literal_eval(line)
            if record["asin"] not in wanted:
                continue
            asin2sentence[record["asin"]] = "".join(feature_to_sentence(record[key]) for key in META_FEATURES if key in record)

    missing = len(wanted) - len(asin2sentence)
    if missing:
        print(f"Warning: {missing} of {len(wanted)} items have no metadata; using empty sentences.")
    return [asin2sentence.get(asin, "") for asin in id2item[1:]]


def encode_sentences(sentences, model_name, batch_size, device):
    """Encode item sentences with a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    print(f"Encoding {len(sentences)} sentences with {model_name}")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(sentences, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, device=device)
    return embeddings.astype(np.float32)


# =========================================================
# Entry point
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Amazon Reviews 2014 for RPG")
    parser.add_argument("--category", type=str, default="Beauty", help=f"Amazon 2014 category. RPG paper uses: {', '.join(RPG_CATEGORIES)}")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory holding raw/ and processed/")
    parser.add_argument("--sent_emb_model", type=str, default="sentence-transformers/sentence-t5-base", help="sentence-transformers model used to embed item metadata")
    parser.add_argument("--batch_size", type=int, default=512, help="Sentence encoding batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device used for sentence encoding")
    parser.add_argument("--no_download", action="store_true", help="Use already-downloaded raw files")
    parser.add_argument("--overwrite", action="store_true", help="Redo every step even if outputs exist")
    return parser.parse_args()


def main(args):
    raw_dir = os.path.join(args.data_dir, "raw")
    out_dir = os.path.join(args.data_dir, "processed", args.category)
    os.makedirs(out_dir, exist_ok=True)

    reviews_path = os.path.join(raw_dir, f"reviews_{args.category}_5.json.gz")
    meta_path = os.path.join(raw_dir, f"meta_{args.category}.json.gz")
    if not args.no_download:
        download_file(f"{SNAP_URL}/reviews_{args.category}_5.json.gz", reviews_path, args.overwrite)
        download_file(f"{SNAP_URL}/meta_{args.category}.json.gz", meta_path, args.overwrite)

    inter_path = os.path.join(out_dir, "inter.json")
    mapping_path = os.path.join(out_dir, "id_mapping.json")
    if args.overwrite or not os.path.exists(inter_path):
        inter, id_mapping = build_sequences(reviews_path)
        with open(inter_path, "w") as f:
            json.dump(inter, f)
        with open(mapping_path, "w") as f:
            json.dump(id_mapping, f)
    else:
        print(f"Reusing {inter_path}")
        with open(mapping_path, "r") as f:
            id_mapping = json.load(f)

    sentences_path = os.path.join(out_dir, "item_sentences.json")
    if args.overwrite or not os.path.exists(sentences_path):
        sentences = build_sentences(meta_path, id_mapping["id2item"])
        with open(sentences_path, "w") as f:
            json.dump(sentences, f)
    else:
        print(f"Reusing {sentences_path}")
        with open(sentences_path, "r") as f:
            sentences = json.load(f)

    emb_path = os.path.join(out_dir, "item_embeddings.npy")
    if args.overwrite or not os.path.exists(emb_path):
        np.save(emb_path, encode_sentences(sentences, args.sent_emb_model, args.batch_size, args.device))
    else:
        print(f"Reusing {emb_path}")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main(parse_args())
