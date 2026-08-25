"""RPG on Amazon Reviews 2014.

RPG tokenizes each item into ``n_codebook`` **unordered** codes with OPQ and
trains a small causal transformer to predict all of them in a single forward
pass. Because no digit refines another there is no beam search over digits,
which is what lets semantic IDs grow to 32 or 64 tokens. Retrieval walks an
item-item similarity graph derived from the learned codebooks, so only a
fraction of the catalog is ever scored.

Reference: Hou et al., "Generating Long Semantic IDs in Parallel for
Recommendation", KDD 2025. https://arxiv.org/abs/2506.05781

Run modes (``--mode``)
----------------------
``preprocess``
    Download Amazon Reviews 2014 and encode item metadata into sentence
    embeddings (delegates to ``data/amazon-2014/preprocess_amazon_2014.py``).
``tokenize``
    Fit OPQ on the item embeddings and write the semantic IDs.
``train``
    Train the model, selecting the checkpoint by validation NDCG@10.
``test``
    Load the checkpoint and report metrics with graph-constrained decoding.
``all``
    ``preprocess`` -> ``tokenize`` -> ``train`` -> ``test`` (default).

Results
-------
Beauty, defaults below (m=32, lr=0.01, tau=0.03, b=20, k=200, q=3), early
stopped at epoch 91 on a single L20:

===================  ======  ======  ======  =======
Setting              R@5     N@5     R@10    N@10
===================  ======  ======  ======  =======
Paper (OpenAI emb.)  0.0550  0.0381  0.0809  0.0464
This example         0.0503  0.0350  0.0728  0.0423
  graph-constrained  0.0498  0.0348  0.0721  0.0420
===================  ======  ======  ======  =======

The gap comes from the item encoder: the paper embeds item metadata with
OpenAI ``text-embedding-3-large`` (3072-d, PCA to 512), while this example
defaults to ``sentence-t5-base`` (768-d, PCA to 128) so it runs without an API
key. Graph-constrained decoding costs about 1% relative NDCG while scoring
6,000 of the 12,101 items per query.

Examples
--------
Reproduce the paper's Beauty setting::

    python run_rpg_amazon_2014.py --category Beauty --device cuda

Other categories, with the hyperparameters from the official repository::

    python run_rpg_amazon_2014.py --category Sports_and_Outdoors \
        --lr 0.003 --n_codebook 16 --num_beams 100 --n_edges 30 --propagation_steps 5
    python run_rpg_amazon_2014.py --category Toys_and_Games \
        --lr 0.003 --n_codebook 16 --num_beams 200 --n_edges 20 --propagation_steps 3
    python run_rpg_amazon_2014.py --category CDs_and_Vinyl \
        --lr 0.001 --n_codebook 64 --num_beams 20 --n_edges 500 --propagation_steps 5
"""

import argparse
import importlib.util
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from torch_rechub.models.generative import RPGModel
from torch_rechub.trainers import RPGTrainer
from torch_rechub.utils.data import RPGSeqDataset
from torch_rechub.utils.opq import OPQTokenizer

RUN_MODES = ("preprocess", "tokenize", "train", "test", "all")


def data_dir(args):
    return os.path.join(SCRIPT_DIR, "data", "amazon-2014")


def processed_dir(args):
    return os.path.join(data_dir(args), "processed", args.category)


def semantic_ids_path(args):
    return os.path.join(processed_dir(args), f"semantic_ids_opq{args.n_codebook}x{args.codebook_size}.json")


def load_inter(args):
    """Load ``{user_id: [item_id, ...]}`` produced by preprocessing."""
    with open(os.path.join(processed_dir(args), "inter.json"), "r") as f:
        return json.load(f)


# =========================================================
# Preprocess
# =========================================================
def preprocess(args):
    """Run the dataset script that lives next to the raw data."""
    path = os.path.join(data_dir(args), "preprocess_amazon_2014.py")
    spec = importlib.util.spec_from_file_location("preprocess_amazon_2014", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main(argparse.Namespace(
        category=args.category,
        data_dir=data_dir(args),
        sent_emb_model=args.sent_emb_model,
        batch_size=args.sent_emb_batch_size,
        device=args.device,
        no_download=False,
        overwrite=False,
    ))


# =========================================================
# Tokenize
# =========================================================
def tokenize(args):
    """Quantize the item embeddings into unordered semantic IDs.

    Only items that appear in a training prefix train the quantizer, so
    validation- and test-only items cannot shape the codebooks. Every item is
    still encoded.
    """
    embeddings = np.load(os.path.join(processed_dir(args), "item_embeddings.npy"))
    inter = load_inter(args)

    train_mask = np.zeros(len(embeddings), dtype=bool)
    for sequence in inter.values():
        for item_id in sequence[:-2]:
            train_mask[item_id - 1] = True
    print(f"OPQ training items: {int(train_mask.sum())} of {len(embeddings)}")

    tokenizer = OPQTokenizer(n_codebook=args.n_codebook, codebook_size=args.codebook_size, pca_dim=args.sent_emb_pca)
    tokenizer.fit(embeddings, train_mask)
    tokenizer.save(semantic_ids_path(args))
    print(f"Semantic IDs saved to {semantic_ids_path(args)} (collision rate {tokenizer.collision_rate():.4f})")


# =========================================================
# Model / data plumbing
# =========================================================
def build_model(args):
    tokenizer = OPQTokenizer.load(semantic_ids_path(args))
    model = RPGModel(
        tokenizer.item_tokens(),
        codebook_size=args.codebook_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        max_seq_len=args.max_seq_len,
        resid_pdrop=args.resid_pdrop,
        embd_pdrop=args.embd_pdrop,
        attn_pdrop=args.attn_pdrop,
        temperature=args.temperature,
    )
    print(f"items={model.n_items - 1} semantic-id length={model.n_digit} params={sum(p.numel() for p in model.parameters()):,}")
    return model


def build_loader(args, inter, mode):
    dataset = RPGSeqDataset(inter, max_seq_len=args.max_seq_len, mode=mode)
    batch_size = args.batch_size if mode == "train" else args.eval_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=mode == "train", num_workers=args.num_workers, collate_fn=RPGSeqDataset.collate_fn)


def build_trainer(args, model):
    return RPGTrainer(
        model,
        optimizer_params={
            "lr": args.lr,
            "weight_decay": args.weight_decay
        },
        n_epoch=args.epochs,
        earlystop_patience=args.patience,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        model_path=args.output_dir,
    )


# =========================================================
# Train / test
# =========================================================
def train(args):
    inter = load_inter(args)
    trainer = build_trainer(args, build_model(args))
    best = trainer.fit(build_loader(args, inter, "train"), build_loader(args, inter, "valid"))
    print("======================================================")
    print("Best validation:", {k: round(v, 4) for k, v in best.items()})
    print("======================================================")


def test(args):
    ckpt = os.path.join(args.output_dir, "model.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Train first or pass --output_dir.")

    model = build_model(args)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    trainer = build_trainer(args, model)
    test_loader = build_loader(args, load_inter(args), "test")

    print("Building the decoding graph...")
    model.build_decoding_graph(n_edges=args.n_edges, chunk_size=args.chunk_size)
    graph = trainer.evaluate(test_loader, use_graph=True, num_beams=args.num_beams, propagation_steps=args.propagation_steps)
    exhaustive = trainer.evaluate(test_loader)

    print("======================================================")
    print("Test (graph-constrained):", {k: round(v, 4) for k, v in graph.items()})
    print("Test (full ranking):     ", {k: round(v, 4) for k, v in exhaustive.items()})
    print(f"Distinct items scored per query: {model.n_visited_items:.0f} of {model.n_items - 1}")
    print("======================================================")
    return graph


# =========================================================
# Argument parsing & dispatch
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RPG on Amazon Reviews 2014")
    parser.add_argument("--mode", type=str, default="all", choices=RUN_MODES, help="Which stage(s) to run")
    parser.add_argument("--category", type=str, default="Beauty", help="Amazon Reviews 2014 category")
    parser.add_argument("--output_dir", type=str, default=None, help="Checkpoint directory (default: ./ckpt/rpg_<category>)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    # item tokenization
    parser.add_argument("--sent_emb_model", type=str, default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--sent_emb_batch_size", type=int, default=512)
    parser.add_argument("--sent_emb_pca", type=int, default=128, help="Whitened-PCA dimension before OPQ, 0 to disable")
    parser.add_argument("--n_codebook", type=int, default=32, help="Semantic-id length m")
    parser.add_argument("--codebook_size", type=int, default=256, help="Codes per digit, must be a power of two")

    # backbone
    parser.add_argument("--n_embd", type=int, default=448)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_inner", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--resid_pdrop", type=float, default=0.0)
    parser.add_argument("--embd_pdrop", type=float, default=0.5)
    parser.add_argument("--attn_pdrop", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.03)

    # optimization
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # graph-constrained decoding
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--n_edges", type=int, default=200)
    parser.add_argument("--propagation_steps", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=1024, help="Rows scored at a time while building the graph")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "ckpt", f"rpg_{args.category}")
    return args


def main(args):
    if args.mode in ("preprocess", "all"):
        preprocess(args)
    if args.mode in ("tokenize", "all"):
        tokenize(args)
    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("test", "all"):
        test(args)


if __name__ == "__main__":
    main(parse_args())
