import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -------------------------
# Long-tail + label-noise dataset
# -------------------------

class NoisyLabelDataset(Dataset):
    """Wraps a torchvision dataset and returns noisy and clean labels."""

    def __init__(self, base_dataset, indices: List[int], noisy_targets: List[int], clean_targets: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
        self.noisy_targets = noisy_targets
        self.clean_targets = clean_targets

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        x, _ = self.base_dataset[original_idx]
        return x, int(self.noisy_targets[idx]), int(self.clean_targets[idx])


def compute_exponential_class_counts(
    max_count: int,
    num_classes: int,
    imbalance_factor: float,
    min_count: int = 1,
) -> List[int]:
    """
    Exponential long-tail allocation.

    Rank 0 keeps max_count samples.
    Rank num_classes - 1 keeps about max_count * imbalance_factor samples.
    """
    if not 0 < imbalance_factor <= 1:
        raise ValueError("imbalance_factor must be in (0, 1]. Use 1.0 for balanced data.")

    if num_classes == 1:
        return [max_count]

    counts = []
    for rank in range(num_classes):
        exponent = rank / (num_classes - 1)
        count = int(max_count * (imbalance_factor ** exponent))
        counts.append(max(min_count, count))
    return counts




def make_long_tailed_noisy_dataset(
    base_dataset,
    num_classes: int,
    imbalance_factor: float,
    noise_rate: float,
    seed: int,
) -> Tuple[NoisyLabelDataset, Dict]:
    """Creates an exponential long-tailed dataset and applies symmetric label noise."""
    if not 0 <= noise_rate <= 1:
        raise ValueError("noise_rate must be in [0, 1].")

    rng = np.random.default_rng(seed)
    targets = np.array(base_dataset.targets)
    class_order = rng.permutation(num_classes)

    class_indices = {c: np.where(targets == c)[0] for c in range(num_classes)}
    max_count = min(len(class_indices[c]) for c in range(num_classes))
    counts_per_rank = compute_exponential_class_counts(max_count, num_classes, imbalance_factor)

    indices_kept: List[int] = []
    clean_targets_kept: List[int] = []
    class_counts: Dict[int, int] = {}
    class_ranks: Dict[int, int] = {}

    for rank, cls in enumerate(class_order):
        cls = int(cls)
        keep_count = counts_per_rank[rank]
        class_counts[cls] = keep_count
        class_ranks[cls] = rank

        selected = rng.choice(class_indices[cls], size=keep_count, replace=False)
        indices_kept.extend(selected.tolist())
        clean_targets_kept.extend([cls] * keep_count)

    indices_kept = np.array(indices_kept)
    clean_targets_kept = np.array(clean_targets_kept)
    perm = rng.permutation(len(indices_kept))

    indices_kept = indices_kept[perm].tolist()
    clean_targets_kept = clean_targets_kept[perm]
    noisy_targets = clean_targets_kept.copy()

    num_noisy = int(noise_rate * len(noisy_targets))
    noisy_indices = rng.choice(len(noisy_targets), size=num_noisy, replace=False)

    for idx in noisy_indices:
        true_label = int(noisy_targets[idx])
        candidate_labels = list(range(num_classes))
        candidate_labels.remove(true_label)
        noisy_targets[idx] = int(rng.choice(candidate_labels))

    dataset = NoisyLabelDataset(
        base_dataset=base_dataset,
        indices=indices_kept,
        noisy_targets=noisy_targets.tolist(),
        clean_targets=clean_targets_kept.tolist(),
    )

    metadata = {
        "imbalance_mode": "exponential",
        "imbalance_factor": imbalance_factor,
        "class_order": class_order.tolist(),
        "class_counts": class_counts,
        "class_ranks": class_ranks,
        "num_train_samples": len(dataset),
        "num_noisy_samples": num_noisy,
    }
    return dataset, metadata


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
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_transform = train_transform
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    elif name == "cifar10":
        in_channels = 3
        num_classes = 10
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    elif name == "cifar100":
        in_channels = 3
        num_classes = 100
        normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    else:
        raise ValueError(f"Unsupported dataset: {name}. Choose from: mnist, cifar10, cifar100.")

    return train_dataset, test_dataset, in_channels, num_classes


# -------------------------
# Models
# -------------------------

class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
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
        return self.classifier(torch.flatten(self.features(x), 1))


def get_model(model_name: str, in_channels: int, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "smallcnn":
        return SmallCNN(in_channels=in_channels, num_classes=num_classes)

    if model_name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None, num_classes=num_classes)
    else:
        raise ValueError("Unsupported model. Choose from: smallcnn, resnet18, resnet34.")

    # CIFAR-style ResNet stem: better for 32x32 images than the ImageNet 7x7 stride-2 stem.
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -------------------------
# Training / evaluation
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct_noisy = 0
    total_samples = 0

    for x, y_noisy, _ in loader:
        x = x.to(device, non_blocking=True)
        y_noisy = y_noisy.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y_noisy)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct_noisy += (logits.argmax(dim=1) == y_noisy).sum().item()
        total_samples += x.size(0)

    return {
        "train_loss": total_loss / total_samples,
        "train_noisy_acc": total_correct_noisy / total_samples,
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> Dict:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in loader:
        if len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch

        logits = model(x.to(device, non_blocking=True))
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(y.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    results = {
        "test_acc": accuracy_score(all_targets, all_preds),
        #"confusion_matrix_json": json.dumps(confusion_matrix(all_targets, all_preds, labels=list(range(num_classes))).tolist()),
    }

    for c in range(num_classes):
        mask = all_targets == c
        results[f"class_{c}_acc"] = float((all_preds[mask] == c).mean()) if mask.sum() else 0.0

    return results


def run_single_experiment(args, seed: int, imbalance_factor: float, noise_rate: float, device: torch.device) -> Dict:
    set_seed(seed)
    train_base, test_dataset, in_channels, num_classes = get_dataset(args.dataset, args.data_dir)

    train_dataset, metadata = make_long_tailed_noisy_dataset(
        base_dataset=train_base,
        num_classes=num_classes,
        imbalance_factor=imbalance_factor,
        noise_rate=noise_rate,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = get_model(args.model, in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_stats = evaluate(model, test_loader, device, num_classes)

        if not args.quiet:
            print(
                f"dataset={args.dataset} model={args.model} seed={seed} "
                f"IF={imbalance_factor} noise={noise_rate} epoch={epoch:03d} "
                f"loss={train_stats['train_loss']:.4f} "
                f"train_noisy_acc={train_stats['train_noisy_acc']:.4f} "
                f"test_acc={test_stats['test_acc']:.4f}",
                flush=True,
            )

    final_stats = evaluate(model, test_loader, device, num_classes)

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": seed,
        "imbalance_mode": "exponential",
        "imbalance_factor": imbalance_factor,
        "noise_rate": noise_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_classes": num_classes,
        "num_train_samples": metadata["num_train_samples"],
        "num_noisy_samples": metadata["num_noisy_samples"],
        **final_stats,
    }

    for c in range(num_classes):
        result[f"class_{c}_train_count"] = metadata["class_counts"][c]
        result[f"class_{c}_rank"] = metadata["class_ranks"][c]

    return result


# -------------------------
# CSV logging
# -------------------------

def make_output_path(args) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_csv:
        filename = f"{args.output_csv}.csv"
        return output_dir / filename

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.dataset}_{args.model}_longtail_noise_{timestamp}.csv"
    return output_dir / filename


def append_result_to_csv(result: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    # Appending each experiment result immediately avoids losing completed runs.
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
        f.flush()
        os.fsync(f.fileno())


def results_to_long_class_df(results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    rows = []
    for _, row in results_df.iterrows():
        for c in range(num_classes):
            rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": row["seed"],
                "imbalance_factor": row["imbalance_factor"],
                "noise_rate": row["noise_rate"],
                "class_id": c,
                "class_rank": row[f"class_{c}_rank"],
                "train_count": row[f"class_{c}_train_count"],
                "class_acc": row[f"class_{c}_acc"],
            })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run long-tail + symmetric label-noise image-classification experiments.")

    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional filename. If omitted, a unique timestamped CSV is created.")

    parser.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--noise_rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--imbalance_factors", type=float, nargs="+", default=[1.0, 0.5, 0.1, 0.05, 0.01, 0.005])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])


    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--save_class_csv", action="store_true", help="Also save a long-format per-class CSV after the full grid finishes.")

    args = parser.parse_args()
    args.pin_memory = not args.no_pin_memory
    return args


def main() -> None:
    args = parse_args()

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    output_path = make_output_path(args)
    print(f"Using device: {device}")
    print(f"Writing experiment results to: {output_path}", flush=True)

    completed_results = []
    for seed in args.seeds:
        for imbalance_factor in args.imbalance_factors:
            for noise_rate in args.noise_rates:
                result = run_single_experiment(args, seed, imbalance_factor, noise_rate, device)
                append_result_to_csv(result, output_path)
                completed_results.append(result)
                print(
                    f"Saved: seed={seed}, IF={imbalance_factor}, noise={noise_rate}, "
                    f"test_acc={result['test_acc']:.4f}",
                    flush=True,
                )

    if args.save_class_csv and completed_results:
        results_df = pd.DataFrame(completed_results)
        class_df = results_to_long_class_df(results_df, num_classes=int(results_df["num_classes"].iloc[0]))
        class_csv_path = output_path.with_name(output_path.stem + "_class_results.csv")
        class_df.to_csv(class_csv_path, index=False)
        print(f"Wrote per-class results to: {class_csv_path}")


if __name__ == "__main__":
    main()
