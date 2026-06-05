import os
import random
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix # Added confusion_matrix here
)


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Long-tail + label noise dataset
# -------------------------

class NoisyLabelDataset(Dataset):
    """
    Wraps a torchvision dataset and replaces labels with noisy labels.
    """

    def __init__(self, base_dataset, indices, noisy_targets, clean_targets):
        self.base_dataset = base_dataset
        self.indices = indices
        self.noisy_targets = noisy_targets
        self.clean_targets = clean_targets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x, _ = self.base_dataset[original_idx]
        y_noisy = self.noisy_targets[idx]
        y_clean = self.clean_targets[idx]
        return x, y_noisy, y_clean


def make_long_tailed_noisy_dataset(
    base_dataset,
    num_classes: int,
    imbalance_type,
    imbalance_factor,
    noise_rate: float,
    seed: int,
):

    rng = np.random.default_rng(seed)

    targets = np.array(base_dataset.targets)
    class_order = rng.permutation(num_classes)

    indices_kept = []
    clean_targets_kept = []

    # Count original samples per class
    class_indices = {
        c: np.where(targets == c)[0] for c in range(num_classes)
    }

    max_count = min(len(class_indices[c]) for c in range(num_classes))

    # Power-law / exponential long-tail allocation
    # Class rank 0 keeps max_count
    # Class rank num_classes-1 keeps max_count * imbalance_ratio
    class_counts = {}


    def compute_class_counts(
        max_count,
        num_classes,
        imbalance_type="exponential",
        imbalance_factor=0.5,
        min_count=1,
    ):
        counts = []

        if imbalance_type == "balanced":
            counts = [max_count] * num_classes

        elif imbalance_type == "exponential":
            for rank in range(num_classes):
                exponent = rank / (num_classes - 1)
                count = int(max_count * (imbalance_factor ** exponent))
                count = max(min_count, count)
                counts.append(count)

        elif imbalance_type == "linear":
            step = imbalance_factor
            for rank in range(num_classes):
                count = int(max_count - step * rank)
                count = max(min_count, count)
                counts.append(count)

        else:
            raise ValueError("Unknown imbalance type")

        return counts



    counts_per_rank = compute_class_counts(
        max_count=max_count,
        num_classes=num_classes,
        imbalance_type=imbalance_type,
        imbalance_factor=imbalance_factor,
    )
    class_ranks = {}
    for rank, cls in enumerate(class_order):
        keep_count = counts_per_rank[rank]
        class_counts[int(cls)] = keep_count
        class_ranks[int(cls)] = rank

        selected = rng.choice(
            class_indices[cls],
            size=keep_count,
            replace=False
        )

        indices_kept.extend(selected.tolist())
        clean_targets_kept.extend([int(cls)] * keep_count)


    # Shuffle retained dataset
    indices_kept = np.array(indices_kept)
    clean_targets_kept = np.array(clean_targets_kept)

    perm = rng.permutation(len(indices_kept))

    indices_kept = indices_kept[perm].tolist()
    clean_targets_kept = clean_targets_kept[perm]

    noisy_targets = clean_targets_kept.copy()

    # Apply symmetric label noise
    num_noisy = int(noise_rate * len(noisy_targets))
    noisy_indices = rng.choice(len(noisy_targets), size=num_noisy, replace=False)

    for idx in noisy_indices:
        true_label = noisy_targets[idx]
        possible_labels = list(range(num_classes))
        possible_labels.remove(int(true_label))
        noisy_targets[idx] = rng.choice(possible_labels)

    dataset = NoisyLabelDataset(
        base_dataset=base_dataset,
        indices=indices_kept,
        noisy_targets=noisy_targets.tolist(),
        clean_targets=clean_targets_kept.tolist(),
    )

    metadata = {
        "class_order": class_order.tolist(),
        "class_counts": class_counts,
        "class_ranks": class_ranks,
        "num_train_samples": len(dataset),
        "num_noisy_samples": num_noisy,
    }

    return dataset, metadata


# -------------------------
# Models
# -------------------------

class SmallCNN(nn.Module):
    """
    Works for MNIST and CIFAR-10.
    Set in_channels=1 for MNIST, 3 for CIFAR.
    """

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -------------------------
# Dataset loading
# -------------------------

def get_dataset(name: str, data_dir: str):
    name = name.lower()

    if name == "mnist":
        in_channels = 1
        num_classes = 10

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )

    elif name == "cifar10":
        in_channels = 3
        num_classes = 10

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_dataset, test_dataset, in_channels, num_classes


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

    macro_f1 = f1_score(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        all_targets,
        all_preds,
        average="micro",
        zero_division=0,
    )

    macro_precision = precision_score(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )

    # -------------------------
    # Per-class accuracy
    # -------------------------

    per_class_accuracy = {}

    for c in range(num_classes):

        mask = (all_targets == c)

        if mask.sum() == 0:
            class_acc = 0.0
        else:
            class_acc = (all_preds[mask] == c).mean()

        per_class_accuracy[c] = float(class_acc)

    # confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    results = {
        "test_acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
    }

    # add per-class accuracies
    for c, v in per_class_accuracy.items():
        results[f"class_{c}_acc"] = v

    return results


import matplotlib.pyplot as plt
def print_and_plot_class_distribution(metadata, title="Training class distribution"):
    class_counts = metadata["class_counts"]

    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    print("\nClass count distribution:")
    for c, count in zip(classes, counts):
        print(f"Class {c}: {count}")

    plt.figure(figsize=(8, 4))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Number of training samples")
    plt.title(title)
    plt.xticks(classes)
    plt.tight_layout()
    plt.show()


def run_single_experiment(
    dataset_name,
    data_dir,
    imbalance_type,
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
        imbalance_type=imbalance_type,
        imbalance_factor=imbalance_factor,
        noise_rate=noise_rate,
        seed=seed,
    )

    if False:
        print_and_plot_class_distribution(
            metadata,
            title=(
                f"{dataset_name.upper()} | "
                f"{imbalance_type} imbalance | "
                f"factor={imbalance_factor} | "
                f"noise={noise_rate}"
            )
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

    model = SmallCNN(
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(device)

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
            f"MacroF1={test_stats['macro_f1']:.4f} | "
            f"MicroF1={test_stats['micro_f1']:.4f}"
        )
        #for c in range(num_classes):
            #print(f"  Class {c} Acc: "f"{test_stats[f'class_{c}_acc']:.4f}")

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


def plot_accuracy_vs_class_count(result):
    class_counts = []
    class_accs = []

    for c in range(10):

        count = result[f"class_{c}_train_count"]
        acc = result[f"class_{c}_acc"]

        class_counts.append(count)
        class_accs.append(acc)

    plt.figure(figsize=(6, 4))

    plt.scatter(class_counts, class_accs)

    for c in range(10):
        plt.annotate(str(c), (class_counts[c], class_accs[c]))

    plt.xscale("log")

    plt.xlabel("Training samples per class (log scale)")
    plt.ylabel("Per-class accuracy")
    plt.title("Accuracy vs class frequency")

    plt.tight_layout()
    plt.show()


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


def plot_avg_accuracy_by_rank(avg_df, noise_rate=0.0, savepath="figures/tmp"):
    df = avg_df[avg_df["noise_rate"] == noise_rate].copy()

    plt.figure(figsize=(7, 4))

    for imbalance_ratio in sorted(df["imbalance_ratio"].unique()):
        sub = df[df["imbalance_ratio"] == imbalance_ratio]
        sub = sub.sort_values("class_rank")

        plt.plot(
            sub["class_rank"],
            sub["mean_class_acc"],
            marker="o",
            label=f"IF={imbalance_ratio}",
        )

        plt.fill_between(
            sub["class_rank"],
            sub["mean_class_acc"] - sub["std_class_acc"].fillna(0),
            sub["mean_class_acc"] + sub["std_class_acc"].fillna(0),
            alpha=0.15,
        )

    plt.xlabel("Class rank")
    plt.ylabel("Mean per-class accuracy")
    plt.title(f"Per-class accuracy by long-tail rank. Noise Rate: {noise_rate}")
    plt.xticks(range(10))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.savefig(savepath)


def plot_avg_accuracy_by_count(avg_df, noise_rate=0.0, savepath="figures/tmp"):
    df = avg_df[avg_df["noise_rate"] == noise_rate].copy()

    plt.figure(figsize=(7, 4))

    for imbalance_ratio in sorted(df["imbalance_ratio"].unique()):
        sub = df[df["imbalance_ratio"] == imbalance_ratio]
        sub = sub.sort_values("mean_train_count", ascending=False)

        plt.plot(
            sub["mean_train_count"],
            sub["mean_class_acc"],
            marker="o",
            label=f"IF={imbalance_ratio}",
        )

    plt.xscale("log")
    plt.xlabel("Mean training samples per class")
    plt.ylabel("Mean per-class accuracy")
    plt.title(f"Per-class accuracy vs class frequency. Noise Rate: {noise_rate}")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.savefig(savepath)
    
    
    
    
    

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

dataset_name = "mnist"
data_dir = "./data"

epochs = 10
batch_size = 128
lr = 1e-3
weight_decay = 1e-4

noise_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
imbalance_ratios = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
seeds = [1, 2, 3]


all_results = []

for seed in seeds:
    for imbalance_ratio in imbalance_ratios:
        for noise_rate in noise_rates:
            result = run_single_experiment(
                dataset_name=dataset_name,
                data_dir=data_dir,
                imbalance_type="exponential",
                imbalance_factor=imbalance_ratio,
                noise_rate=noise_rate,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
            )
            #plot_accuracy_vs_class_count(result)
            all_results.append(result)


results_df = pd.DataFrame(all_results)
results_df
results_df.to_csv("longtail_noise_results.csv", index=False)

class_results_df = results_to_long_class_df(results_df, num_classes=10)
class_results_df.to_csv("longtail_class_results.csv", index=False)
class_results_df

avg_class_results = (
    class_results_df
    .groupby(["imbalance_ratio", "noise_rate", "class_rank"], as_index=False)
    .agg( mean_train_count=("train_count", "mean"),
        std_train_count=("train_count", "std"),
        mean_class_acc=("class_acc", "mean"),
        std_class_acc=("class_acc", "std"),)
)
avg_class_results


#plot_avg_accuracy_by_rank(avg_class_results, noise_rate=0.0)
#plot_avg_accuracy_by_count(avg_class_results, noise_rate=0.0)
#plot_avg_accuracy_by_rank(avg_class_results, noise_rate=0.4)
#plot_avg_accuracy_by_count(avg_class_results, noise_rate=0.4)

