import argparse
import json
import os

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, T5Config, T5ForConditionalGeneration, T5Tokenizer

from torch_rechub.models.generative.tiger import TIGERModel
from torch_rechub.utils.data import TigerSeqDataset, Trie


def train(args):
    """
    Training example of TIGER on Amazon Books dataset.
    
    Parameters
    ----------
    args : argparse.Namespace
        The arguments for training and testing.More details of the arguments can be found in the parse_*_args functions.
    """
    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(args.data_inter_path, 'r') as f:
        inters_json = json.load(f)
    with open(args.data_indice_path, 'r') as f:
        indices_json = json.load(f)
    train_data = TigerSeqDataset(inters_json, indices_json, args.max_his_len, mode="train")
    valid_data = TigerSeqDataset(inters_json, indices_json, args.max_his_len, mode="valid")
    config.vocab_size = len(tokenizer)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    model = TIGERModel(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

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
        tokenizer=tokenizer,
        data_collator=valid_data.get_collate_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint,)

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


def parse_global_args(parser):
    parser.add_argument("--base_model", type=str, default="t5-small", help="basic model path")
    parser.add_argument("--output_dir", type=str, default="./ckpt", help="The output directory")
    return parser


def parse_dataset_args(parser):
    parser.add_argument("--data_inter_path", type=str, default="./data/amazon-books/book.inter.json", help="the user-item interaction file")
    parser.add_argument("--data_indice_path", type=str, default="./data/amazon-books/book.index.json", help="the item indices file")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20, help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False, help="whether add sequential prefix in history")
    return parser


def parse_train_args(parser):
    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)

    return parser


def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str, default="/home/liuwei/workshop/torch-rechub/examples/generative/ckp", help="The checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=True, help="whether filter illegal items")
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1, help="test sample number, -1 represents using all test data")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10", help="test metrics, separate by comma")
    return parser


def generate_data():
    """
    An example of generating Amazon Books dataset for TIGER. 
    The generated dataset will be saved in "./data/amazon-books". 
    The dataset contains two files: "inter.json" and "semantic_ids.json". "inter.json".
    You can obtain "semantic_ids.json" from the RQ-VAE model(run_rqvae_amazon_books.py) or from your own implementation.
    """
    save_dir = "./data/amazon-books"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # inter.json
    # -------------------------
    inter_data = {"0": [1, 2, 3, 4, 1, 2, 3, 4], "1": [2, 3, 4, 1, 2, 3, 4], "2": [3, 4, 6, 1, 2, 3]}

    with open(os.path.join(save_dir, "inter.json"), "w") as f:
        json.dump(inter_data, f, indent=4)

    # -------------------------
    # semantic_ids.json
    # -------------------------
    index_data = {"1": ["<a-1>", "<b-10>"], "2": ["<a-1>", "<b-20>"], "3": ["<a-2>", "<b-30>"], "4": ["<a-2>", "<b-40>"], "5": ["<a-3>", "<b-50>"], "6": ["<a-3>", "<b-60>"], "7": ["<a-4>", "<b-70>"]}

    with open(os.path.join(save_dir, "semantic_ids.json"), "w") as f:
        json.dump(index_data, f, indent=4)

    print("book dataset generated.")


def test(args):
    """
    Testing example of TIGER on Amazon Books dataset.
    
    Parameters
    ----------
    args : argparse.Namespace
        The arguments for training and testing. More details of the arguments can be found in the parse_*_args functions.
    """
    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    with open(args.data_inter_path, 'r') as f:
        inters_json = json.load(f)
    with open(args.data_indice_path, 'r') as f:
        indices_json = json.load(f)
    test_data = TigerSeqDataset(inters_json, indices_json, args.max_his_len, mode="test")
    add_num = tokenizer.add_tokens(test_data.get_new_tokens())
    config.vocab_size = len(tokenizer)
    print("add {} new token.".format(add_num))
    # tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = model = TIGERModel.from_pretrained(args.ckpt_path, low_cpu_mem_usage=True)
    all_items = test_data.get_all_items()
    candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
    prefix_allowed_tokens = candidate_trie.def_prefix_allowed_tokens_fn(candidate_trie)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=test_data.get_collate_fn(tokenizer), shuffle=True, num_workers=4, pin_memory=True)
    print("data num:", len(test_data))
    model.eval()
    metrics = args.metrics.split(",")
    with torch.no_grad():

        metrics_results = {}
        total = 0

        for step, batch in enumerate(tqdm(test_loader)):
            inputs = batch["input_ids"]
            targets = batch["labels"]
            total += len(targets)

            output = model.generate(
                input_ids=inputs,
                attention_mask=batch["attention_mask"],
                max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

            output_ids = output["sequences"]
            scores = output["sequences_scores"]

            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            targets_text = tokenizer.batch_decode(targets, skip_special_tokens=True)
            topk_res = get_topk_results(output_text, scores, targets_text, args.num_beams, all_items=all_items if args.filter_items else None)

            batch_metrics_res = get_metrics_results(topk_res, metrics)

            for m, res in batch_metrics_res.items():
                if m not in metrics_results:
                    metrics_results[m] = res
                else:
                    metrics_results[m] += res

        # average the metrics results
        for m in metrics_results:
            metrics_results[m] = metrics_results[m] / total

        print("======================================================")
        print("Test results: ", metrics_results)
        print("======================================================")


import math


def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ", "") for _ in predictions]
    targets = [_.strip().replace(" ", "") for _ in targets]
    # print(predictions)##################
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k:(b + 1) * k]
        batch_scores = scores[b * k:(b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results


def get_topk_ranking_results(predictions, targets, k, all_items=None):
    results = []
    B = len(targets)

    for b in range(B):
        batch_seqs = predictions[b]
        target_item = targets[b]
        one_results = []
        for sorted_pred in predictions:
            if sorted_pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit


if __name__ == "__main__":
    """
    python  run_tiger_amazon_books.py \
    --base_model /t5-small \
    --output_dir ./ckp \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --temperature 1.0
    """
    generate_data()
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()
    train(args)
    test(args)
