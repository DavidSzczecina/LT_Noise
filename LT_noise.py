import os
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from plot import plot_accuracy_vs_class_count, plot_avg_accuracy_by_rank, plot_avg_accuracy_by_count
from models import get_model
from data import get_dataset, make_long_tailed_noisy_dataset

from sklearn.metrics import accuracy_score

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




# -------------------------
# Training / evaluation
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct_noisy = 0
    total_samples = 0

    for x, y_noisy, _ in loader:
        x = x.to(device)
        y_noisy = y_noisy.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y_noisy)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct_noisy += (preds == y_noisy).sum().item()
        total_samples += x.size(0)
    return {
        "train_loss": total_loss / total_samples,
        "train_noisy_acc": total_correct_noisy / total_samples,
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        if len(batch) == 3:
            x, _, y_clean = batch
            y = y_clean
        else:
            x, y = batch
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(y.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    per_class_accuracy = {}
    for c in range(num_classes):
        mask = (all_targets == c)
        if mask.sum() == 0:
            class_acc = 0.0
        else:
            class_acc = (all_preds[mask] == c).mean()
        per_class_accuracy[c] = float(class_acc)
    results = {"test_acc": acc}
    # add per-class accuracies
    for c, v in per_class_accuracy.items():
        results[f"class_{c}_acc"] = v
    return results


def run_single_experiment(
    dataset_name,
    data_dir,
    imbalance_factor,
    noise_rate,
    seed,
    epochs,
    batch_size,
    lr,
    weight_decay,
    device,
):
    set_seed(seed)

    train_base, test_dataset, in_channels, num_classes = get_dataset(dataset_name, data_dir)
    train_dataset, metadata = make_long_tailed_noisy_dataset(
        base_dataset=train_base,
        num_classes=num_classes,
        imbalance_factor=imbalance_factor,
        noise_rate=noise_rate,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = get_model(args.dataset).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        test_stats = evaluate(model, test_loader, device, num_classes=num_classes,)
        print(
            f"Seed={seed} | Imb={imbalance_factor} | Noise={noise_rate} | "
            f"Epoch={epoch:03d} | "
            f"Loss={train_stats['train_loss']:.4f} | "
            f"TrainNoisyAcc={train_stats['train_noisy_acc']:.4f} | "
            f"TestAcc={test_stats['test_acc']:.4f} | "
        )

    final_stats = evaluate(model, test_loader, device, num_classes=num_classes,)

    result = {
        "dataset": dataset_name,
        "seed": seed,
        "imbalance_ratio": imbalance_factor,
        "noise_rate": noise_rate,
        "epochs": epochs,
        "num_train_samples": metadata["num_train_samples"],
        "num_noisy_samples": metadata["num_noisy_samples"],
        **final_stats,
    }
    for c, count in metadata["class_counts"].items():
        result[f"class_{c}_train_count"] = count
        result[f"class_{c}_rank"] = metadata["class_ranks"][c]
    return result


def results_to_long_class_df(results_df, num_classes=10):
    rows = []
    for _, row in results_df.iterrows():
        for c in range(num_classes):
            rows.append({
                "dataset": row["dataset"],
                "seed": row["seed"],
                "imbalance_ratio": row["imbalance_ratio"],
                "noise_rate": row["noise_rate"],
                "class_id": c,
                "class_rank": row[f"class_{c}_rank"],
                "train_count": row[f"class_{c}_train_count"],
                "class_acc": row[f"class_{c}_acc"],
            })
    return pd.DataFrame(rows)



def run_experiment_grid(
    dataset_name,
    data_dir="../datasets",
    output_dir="./results",
    epochs=5,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    noise_rates=None,
    imbalance_ratios=None,
    seeds=None,
):
    if noise_rates is None:
        noise_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if imbalance_ratios is None:
        imbalance_ratios = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"{dataset_name}_summary_results.csv")
    class_path = os.path.join(output_dir, f"{dataset_name}_per_class_results.csv")
    avg_class_path = os.path.join(output_dir, f"{dataset_name}_avg_by_class_rank_results.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")

    num_classes = 100 if dataset_name.lower() == "cifar100" else 10

    for seed in seeds:
        for imbalance_ratio in imbalance_ratios:
            for noise_rate in noise_rates:

                result = run_single_experiment(
                    dataset_name=dataset_name,
                    data_dir=data_dir,
                    imbalance_factor=imbalance_ratio,
                    noise_rate=noise_rate,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device,
                )

                # Append this single experiment immediately
                result_df = pd.DataFrame([result])

                write_header = not os.path.exists(summary_path)
                result_df.to_csv(
                    summary_path,
                    mode="a",
                    header=write_header,
                    index=False,
                )

                print(
                    f"Saved experiment: seed={seed}, "
                    f"IF={imbalance_ratio}, noise={noise_rate}"
                )

                # Rebuild per-class and average files from saved summary so far
                saved_results_df = pd.read_csv(summary_path)

                class_results_df = results_to_long_class_df(
                    saved_results_df,
                    num_classes=num_classes,
                )

                avg_class_results = (
                    class_results_df
                    .groupby(
                        ["dataset", "imbalance_ratio", "noise_rate", "class_rank"],
                        as_index=False,
                    )
                    .agg(
                        mean_train_count=("train_count", "mean"),
                        std_train_count=("train_count", "std"),
                        mean_class_acc=("class_acc", "mean"),
                        std_class_acc=("class_acc", "std"),
                    )
                )

                class_results_df.to_csv(class_path, index=False)
                avg_class_results.to_csv(avg_class_path, index=False)

    results_df = pd.read_csv(summary_path)
    class_results_df = pd.read_csv(class_path)
    avg_class_results = pd.read_csv(avg_class_path)

    print(f"Saved: {summary_path}")
    print(f"Saved: {class_path}")
    print(f"Saved: {avg_class_path}")

    return results_df, class_results_df, avg_class_results




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--noise_rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--imbalance_ratios", type=float, nargs="+", default=[1.0, 0.5, 0.1, 0.05, 0.01, 0.005])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    run_experiment_grid(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        noise_rates=args.noise_rates,
        imbalance_ratios=args.imbalance_ratios,
        seeds=args.seeds,
    )