# data.py
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset



# Long-tail + label noise dataset
class NoisyLabelDataset(Dataset):
    #Wraps a torchvision dataset and replaces labels with noisy labels.
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
    imbalance_factor,
    noise_rate: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    
    #targets = np.array(base_dataset.targets)
    targets = np.asarray(base_dataset.targets, dtype=np.int64)

    class_order = rng.permutation(num_classes)
    
    print(f"Seed {seed}: class_order = {class_order.tolist()}")

    indices_kept = []
    clean_targets_kept = []

    # Count original samples per class
    class_indices = {c: np.where(targets == c)[0] for c in range(num_classes)}
    max_count = min(len(class_indices[c]) for c in range(num_classes))

    # Power-law / exponential long-tail allocation
    # Class rank 0 keeps max_count
    class_counts = {}
    counts_per_rank = []
    min_count=1
    for rank in range(num_classes):
        exponent = rank / (num_classes - 1)
        count = int(max_count * (imbalance_factor ** exponent))
        count = max(min_count, count)
        counts_per_rank.append(count)

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
            download=False,
            transform=train_transform,
        )

        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=False,
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
            download=False,
            transform=train_transform,
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=False,
            transform=test_transform,
        )
    elif name == "cifar100":
        in_channels = 3
        num_classes = 100

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ])

        train_dataset = datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=False,
            transform=train_transform,
        )

        test_dataset = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=False,
            transform=test_transform,
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_dataset, test_dataset, in_channels, num_classes




