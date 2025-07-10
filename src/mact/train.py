import argparse
import torch
import numpy as np
import os
from pathlib import Path
import pickle
from copy import deepcopy
from tqdm import tqdm

from .dataset import DatasetsLoader
from .utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from .policy import ACTPolicy

CKPT_DIR = "ckpt"
EPISODE_LEN = 400
CAMERA_NAMES = ["top"]


def train_model(args: argparse.Namespace) -> None:
    set_seed(1)
    # command line parameters
    batch_size_train = args.batch_size
    batch_size_val = args.batch_size
    num_epochs = args.num_epochs

    # fixed parameters
    state_dim = 7
    lr_backbone = 1e-5
    backbone = "resnet18"
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": lr_backbone,
        "backbone": backbone,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "camera_names": CAMERA_NAMES,
    }

    config = {
        "num_epochs": num_epochs,
        "episode_len": EPISODE_LEN,
        "state_dim": state_dim,
        "lr": args.lr,
        "policy_config": policy_config,
        "seed": args.seed,
        "temporal_agg": args.temporal_agg,
        "camera_names": CAMERA_NAMES,
    }

    datasets_loader = DatasetsLoader(Path(".records"), CAMERA_NAMES, args.batch_size)

    # save dataset stats
    os.makedirs(CKPT_DIR, exist_ok=True)
    stats_path = os.path.join(CKPT_DIR, f"dataset_stats.pkl")
    datasets_loader.norm_stats.dump(stats_path)

    best_ckpt_info = train_bc(datasets_loader.train, datasets_loader.validate, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    seed = config["seed"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                image_data, qpos_data, action_data, is_pad = data
                forward_dict = policy(
                    qpos_data.cuda(),
                    image_data.cuda(),
                    action_data.cuda(),
                    is_pad.cuda(),
                )
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            image_data, qpos_data, action_data, is_pad = data
            forward_dict = policy(
                qpos_data.cuda(), image_data.cuda(), action_data.cuda(), is_pad.cuda()
            )
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)

    ckpt_path = os.path.join(CKPT_DIR, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(CKPT_DIR, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    return best_ckpt_info
