"""RQVAE Model Example on Amazon-Books Dataset."""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from torch_rechub.models.generative.rqvae import RQVAEModel
from torch_rechub.trainers.rqvae_trainer import Trainer
from torch_rechub.utils.data import EmbDataset

sys.path.append("../..")


def parse_args():
    parser = argparse.ArgumentParser(description="Index")
    # General parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
    )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument("--data_path", type=str, default="./data/amazon-books/processed/item_embeddings_tinyllama.pt", help="Input data path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument("--model_path", type=str, default="./", help="output directory for model")

    # RQVAE specific parameters
    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256], help='emb num of every vq')
    parser.add_argument('--prefix', type=str, nargs='+', default=["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"], help='prefix for semantic id')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64], help='hidden sizes of every layer')

    return parser.parse_args()


if __name__ == '__main__':
    """
    example command:
    python run_rqvae_amazon_books.py --data_path ./data/amazon-books/processed/item_embeddings_tinyllama.pt --device cuda:1
    """
    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")
    """build dataset"""
    data = EmbDataset(args.data_path)
    print(f"Dataset size: {len(data)} samples, dimension: {data.dim}")
    model = RQVAEModel(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters
    )
    print(model)
    data_loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(model, optimizer_fn=torch.optim.Adam, optimizer_params={'lr': args.learning_rate, 'weight_decay': 1e-5}, n_epoch=args.epochs, device=args.device, model_path=args.model_path, eval_step=args.eval_step)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)

    #------------------------------generate Semantic IDs-------------------------------------------
    test_data_loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # you can try to use the best model based on loss or collision rate
    # best_model_path = os.path.join(args.model_path, "model_best_loss.pth")
    best_model_path = os.path.join(args.model_path, "model_best_collision_rate.pth")
    state_dict = torch.load(best_model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    all_indices_dict = model.generate_semantic_ids(data, test_data_loader, prefix=args.prefix, use_sk=True, device=args.device)
    with open(os.path.join(args.model_path, "semantic_ids.json"), 'w') as fp:
        json.dump(all_indices_dict, fp)
