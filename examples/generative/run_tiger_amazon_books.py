"""TIGER Model Example on Amazon-Books Dataset.

TIGER trains a T5 seq2seq model **from scratch** to autoregressively generate
the semantic ID of the next item. A semantic ID is a short tuple of codebook
tokens (e.g. ``<a_1><b_10>``) produced by RQ-VAE over item embeddings. Following
the TIGER paper, ``--base_model`` only supplies the T5 *architecture/config* and
tokenizer; the transformer weights are randomly initialized, not loaded from a
pretrained NL checkpoint (the semantic-ID vocabulary is not natural language).

Run modes (``--mode``)
----------------------
``generate-toy-data``
    Write a tiny synthetic ``inter.json`` + ``semantic_ids.json`` to the exact
    paths the loader reads from, so the example is runnable end to end.
``train``
    Register the semantic-id tokens, resize the embedding table, train T5 from
    scratch, and save tokenizer/config/model to ``--output_dir``.
``test``
    Load tokenizer/config/model from ``--ckpt_path`` (defaults to
    ``--output_dir``) and report hit@k / ndcg@k with constrained beam search.
``all``
    ``generate-toy-data`` -> ``train`` -> ``test`` (default).

Examples
--------
Toy end-to-end smoke run::

    python run_tiger_amazon_books.py --mode all --epochs 5 --per_device_batch_size 4

Real data: first obtain ``semantic_ids.json`` from RQ-VAE
(``run_rqvae_amazon_books.py``) and an ``inter.json`` of user item-id
sequences, then::

    python run_tiger_amazon_books.py --mode train \
        --data_inter_path ./data/amazon-books/inter.json \
        --data_indice_path ./data/amazon-books/semantic_ids.json
    python run_tiger_amazon_books.py --mode test --ckpt_path ./ckpt
"""

import argparse
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, T5Config, T5Tokenizer

from torch_rechub.models.generative.tiger import TIGERModel
from torch_rechub.utils.data import TigerSeqDataset, Trie

RUN_MODES = ("generate-toy-data", "train", "test", "all")


# =========================================================
# Toy data
# =========================================================
def generate_toy_data(args):
    """Write a tiny Amazon-Books-style toy dataset.

    The files are written to the *exact* paths the dataset loader reads from
    (``--data_inter_path`` / ``--data_indice_path``), so generated and read
    filenames are guaranteed to match.
    """
    inter_data = {
        "0": [1,
              2,
              3,
              4,
              1,
              2,
              3,
              4],
        "1": [2,
              3,
              4,
              1,
              2,
              3,
              4],
        "2": [3,
              4,
              6,
              1,
              2,
              3],
    }
    index_data = {
        "1": ["<a_1>",
              "<b_10>"],
        "2": ["<a_1>",
              "<b_20>"],
        "3": ["<a_2>",
              "<b_30>"],
        "4": ["<a_2>",
              "<b_40>"],
        "5": ["<a_3>",
              "<b_50>"],
        "6": ["<a_3>",
              "<b_60>"],
        "7": ["<a_4>",
              "<b_70>"],
    }
    _write_json(args.data_inter_path, inter_data)
    _write_json(args.data_indice_path, index_data)
    print("Toy Amazon-Books dataset generated:")
    print(f"  inter        -> {args.data_inter_path}")
    print(f"  semantic_ids -> {args.data_indice_path}")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_inters_and_indices(args):
    with open(args.data_inter_path, "r") as f:
        inters_json = json.load(f)
    with open(args.data_indice_path, "r") as f:
        indices_json = json.load(f)
    return inters_json, indices_json


# =========================================================
# Train
# =========================================================
def train(args):
    """Train TIGER from scratch on semantic-id sequences.

    Per the TIGER paper the T5 encoder-decoder is trained from scratch:
    ``TIGERModel(config)`` builds the architecture from ``--base_model``'s config
    with randomly initialized weights (no pretrained NL checkpoint is loaded,
    since the semantic-ID vocabulary is not natural language). The dataset's
    semantic-id tokens are added to the tokenizer *before* resizing the embedding
    table, so tokens such as ``<a_1>`` map to single ids instead of being split
    into subwords. Tokenizer and config are saved to ``--output_dir`` so ``test``
    reloads the exact vocabulary.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=args.model_max_length)
    config = T5Config.from_pretrained(args.base_model)

    inters_json, indices_json = load_inters_and_indices(args)
    train_data = TigerSeqDataset(inters_json, indices_json, args.max_his_len, mode="train")
    valid_data = TigerSeqDataset(inters_json, indices_json, args.max_his_len, mode="valid")

    # Critical: register the semantic-id tokens and grow the embedding table.
    num_added = tokenizer.add_tokens(train_data.get_new_tokens())
    print(f"Added {num_added} semantic-id tokens (vocab size -> {len(tokenizer)})")
    config.vocab_size = len(tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    model = TIGERModel(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.config.use_cache = False

    # transformers>=5 renamed Trainer's ``tokenizer`` argument to
    # ``processing_class``; pick whichever the installed version accepts.
    import inspect

    tok_kw = "processing_class" if "processing_class" in inspect.signature(transformers.Trainer.__init__).parameters else "tokenizer"

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        ),
        data_collator=valid_data.get_collate_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        **{tok_kw: tokenizer},
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
    print(f"Model saved to {args.output_dir}")


# =========================================================
# Test
# =========================================================
def test(args):
    """Evaluate a trained TIGER checkpoint with constrained beam search.

    Loads tokenizer, config and model from ``--ckpt_path`` (falling back to
    ``--output_dir``) so the semantic-id vocabulary matches what training saved.
    """
    ckpt = args.ckpt_path or args.output_dir
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt}. Train first or pass --ckpt_path.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(ckpt, model_max_length=args.model_max_length)

    inters_json, indices_json = load_inters_and_indices(args)
    sample_num = args.sample_num if args.sample_num and args.sample_num > 0 else 0
    test_data = TigerSeqDataset(inters_json, indices_json, args.max_his_len, mode="test", sample_num=sample_num)

    # Load the model with the checkpoint's OWN config so the embedding size matches
    # the saved weights. Passing a grown vocab_size into from_pretrained would raise
    # an embedding-size mismatch before we ever reached resize_token_embeddings.
    model = TIGERModel.from_pretrained(ckpt, low_cpu_mem_usage=False)

    # Defensive: if the checkpoint tokenizer is missing any semantic-id token
    # (e.g. a vanilla T5 checkpoint), add them and then resize the loaded model.
    num_added = tokenizer.add_tokens(test_data.get_new_tokens())
    if num_added:
        print(f"Checkpoint tokenizer was missing {num_added} semantic-id tokens; added them.")
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.config.use_cache = True  # speed up constrained beam search
    model.eval()

    all_items = test_data.get_all_items()
    candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
    prefix_allowed_tokens = candidate_trie.def_prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=test_data.get_collate_fn(tokenizer),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print("test sample num:", len(test_data))

    metrics = args.metrics.split(",")
    metrics_results = {}
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="testing"):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"]
            total += len(targets)

            output = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

            output_text = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
            targets_text = tokenizer.batch_decode(targets, skip_special_tokens=True)
            topk_res = get_topk_results(output_text, output["sequences_scores"], targets_text, args.num_beams, all_items=all_items if args.filter_items else None)

            for m, res in get_metrics_results(topk_res, metrics).items():
                metrics_results[m] = metrics_results.get(m, 0.0) + res

    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / max(total, 1)

    print("======================================================")
    print("Test results:", metrics_results)
    print("======================================================")
    return metrics_results


# =========================================================
# Metrics
# =========================================================
def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    batch = len(targets)
    predictions = [p.strip().replace(" ", "") for p in predictions]
    targets = [t.strip().replace(" ", "") for t in targets]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(batch):
        batch_seqs = predictions[b * k:(b + 1) * k]
        batch_scores = scores[b * k:(b + 1) * k]
        sorted_pairs = sorted(zip(batch_seqs, batch_scores), key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        results.append([1 if pred == target_item else 0 for pred, _ in sorted_pairs])

    return results


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            res[m] = hit_k(topk_results, int(m.split("@")[1]))
        elif m.lower().startswith("ndcg"):
            res[m] = ndcg_k(topk_results, int(m.split("@")[1]))
        else:
            raise NotImplementedError(f"Unsupported metric: {m}")
    return res


def ndcg_k(topk_results, k):
    """Leave-one-out NDCG@k. Each user has a single ground-truth item, so the
    ideal DCG is 1.0 and NDCG reduces to DCG.
    """
    ndcg = 0.0
    for row in topk_results:
        ndcg += sum(rel / math.log(i + 2, 2) for i, rel in enumerate(row[:k]))
    return ndcg


def hit_k(topk_results, k):
    return float(sum(1 for row in topk_results if sum(row[:k]) > 0))


# =========================================================
# Argument parsing & dispatch
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TIGER on Amazon-Books")
    parser.add_argument("--mode", type=str, default="all", choices=RUN_MODES, help="Which stage(s) to run")

    # global
    parser.add_argument("--base_model", type=str, default="t5-small", help="Base T5 model name or path")
    parser.add_argument("--output_dir", type=str, default="./ckpt/tiger_amazon", help="Directory to save tokenizer/config/checkpoints")

    # dataset
    parser.add_argument("--data_inter_path", type=str, default="./data/amazon-books/inter.json", help="User item-id sequence file")
    parser.add_argument("--data_indice_path", type=str, default="./data/amazon-books/semantic_ids.json", help="Item semantic-id file")
    parser.add_argument("--max_his_len", type=int, default=20, help="Max items in history, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False, help="Reserved: add a sequential prefix in history")

    # train
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)

    # test
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint dir to load for testing (default: --output_dir)")
    parser.add_argument("--filter_items", action="store_true", default=True, help="Filter out illegal generated items")
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of test samples, -1 means use all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10", help="Comma separated test metrics")
    return parser.parse_args()


def main(args):
    if args.mode in ("generate-toy-data", "all"):
        generate_toy_data(args)
    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("test", "all"):
        test(args)


if __name__ == "__main__":
    main(parse_args())
