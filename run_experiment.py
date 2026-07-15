import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from dataset import CustomCIFAR100, transform_test, transform_train
from metrics import compute_zrf, get_membership_attack_prob
from model import ResNet18
from unlearn import (
    blindspot_unlearner,
    blindspot_unlearner_retain_aware,
    fit_one_unlearning_cycle,
)
from utils import evaluate, fit_one_cycle


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CIFAR100 machine-unlearning experiment.")
    parser.add_argument("--data-root", default="data", help="CIFAR100 download/cache directory.")
    parser.add_argument("--out-dir", default="outputs", help="Directory for checkpoints, csv and figures.")
    parser.add_argument("--forget-class", type=int, default=69, help="CIFAR100 fine class to forget.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless you know multiprocessing is safe.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-epochs", type=int, default=5)
    parser.add_argument("--gold-epochs", type=int, default=5)
    parser.add_argument("--amnesiac-epochs", type=int, default=3)
    parser.add_argument("--badteacher-epochs", type=int, default=1)
    parser.add_argument("--retain-ce-weight", type=float, default=1.0)
    parser.add_argument("--retain-subset-ratio", type=float, default=0.3)
    parser.add_argument("--quick", action="store_true", help="Run a small CPU-friendly pipeline check.")
    parser.add_argument("--max-retain-train", type=int, default=None)
    parser.add_argument("--max-forget-train", type=int, default=None)
    parser.add_argument("--max-retain-valid", type=int, default=None)
    parser.add_argument("--max-forget-valid", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained ResNet18 weights.")
    parser.add_argument("--model", choices=["resnet18", "tiny"], default="resnet18")
    parser.add_argument("--synthetic", action="store_true", help="Use random in-memory data for a fast smoke test.")
    parser.add_argument("--skip-gold", action="store_true", help="Skip Gold retrain to save time.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore existing checkpoints and retrain.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit(indices, max_count, rng):
    if max_count is None or len(indices) <= max_count:
        return indices
    return rng.sample(indices, max_count)


class AmnesiacDataset(Dataset):
    def __init__(self, base_dataset, forget_indices, retain_indices, num_classes=20, forget_coarse=19, seed=42):
        self.base_dataset = base_dataset
        self.items = []
        rng = random.Random(seed)
        candidates = [i for i in range(num_classes) if i != forget_coarse]
        for idx in forget_indices:
            self.items.append((idx, rng.choice(candidates)))
        for idx in retain_indices:
            self.items.append((idx, None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, pos):
        idx, replacement = self.items[pos]
        x, fine_label, coarse_label = self.base_dataset[idx]
        if replacement is not None:
            coarse_label = replacement
        return x, fine_label, coarse_label


class SyntheticCIFAR100(Dataset):
    def __init__(self, samples_per_class=4, image_size=32, seed=42):
        generator = torch.Generator().manual_seed(seed)
        self.targets = []
        self.images = []
        for fine_label in range(100):
            coarse_label = fine_label_to_coarse(fine_label)
            for _ in range(samples_per_class):
                image = torch.randn(3, image_size, image_size, generator=generator)
                self.images.append((image, fine_label, coarse_label))
                self.targets.append(fine_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


class TinyCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def fine_label_to_coarse(fine_label):
    coarse_map = {
        0: [4, 30, 55, 72, 95],
        1: [1, 32, 67, 73, 91],
        2: [54, 62, 70, 82, 92],
        3: [9, 10, 16, 28, 61],
        4: [0, 51, 53, 57, 83],
        5: [22, 39, 40, 86, 87],
        6: [5, 20, 25, 84, 94],
        7: [6, 7, 14, 18, 24],
        8: [3, 42, 43, 88, 97],
        9: [12, 17, 37, 68, 76],
        10: [23, 33, 49, 60, 71],
        11: [15, 19, 21, 31, 38],
        12: [34, 63, 64, 66, 75],
        13: [26, 45, 77, 79, 99],
        14: [2, 11, 35, 46, 98],
        15: [27, 29, 44, 78, 93],
        16: [36, 50, 65, 74, 80],
        17: [47, 52, 56, 59, 96],
        18: [8, 13, 48, 58, 90],
        19: [41, 69, 81, 85, 89],
    }
    for coarse_label, fine_labels in coarse_map.items():
        if fine_label in fine_labels:
            return coarse_label
    raise ValueError(f"Unknown CIFAR100 fine label: {fine_label}")


def split_indices(dataset, forget_class, args, rng):
    targets = list(dataset.targets)
    forget = [i for i, y in enumerate(targets) if y == forget_class]
    retain = [i for i, y in enumerate(targets) if y != forget_class]
    return forget, retain


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_or_train(path, train_fn, force_retrain):
    if path.exists() and not force_retrain:
        return False
    train_fn()
    return True


def plot_results(summary, out_dir):
    methods = summary["Method"].tolist()
    x = range(len(methods))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar([i - width / 2 for i in x], summary["Forget_Acc"], width, label="Forget Acc (lower=better)", color="#d1495b")
    ax1.bar([i + width / 2 for i in x], summary["Retain_Acc"], width, label="Retain Acc (higher=better)", color="#2e86ab")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_title("Forget vs Retain accuracy across unlearning methods")
    ax1.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig_accuracy_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar([i - width / 2 for i in x], summary["MIA"], width, label="MIA (lower=better)", color="#edae49")
    ax2.bar([i + width / 2 for i in x], summary["ZRF"], width, label="ZRF (higher=better)", color="#66a182")
    ax2.set_ylabel("Score")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_title("Privacy / forgetting metrics across unlearning methods")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig_privacy_metrics.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.quick:
        args.base_epochs = min(args.base_epochs, 1)
        args.gold_epochs = min(args.gold_epochs, 1)
        args.amnesiac_epochs = min(args.amnesiac_epochs, 1)
        args.badteacher_epochs = min(args.badteacher_epochs, 1)
        args.max_forget_train = args.max_forget_train or 32
        args.max_retain_train = args.max_retain_train or 256
        args.max_forget_valid = args.max_forget_valid or 32
        args.max_retain_valid = args.max_retain_valid or 256
        args.batch_size = min(args.batch_size, 32)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"
    print(f"device={device}, quick={args.quick}, out_dir={out_dir}")

    if args.synthetic:
        train_ds = SyntheticCIFAR100(samples_per_class=4, seed=args.seed)
        valid_ds = SyntheticCIFAR100(samples_per_class=2, seed=args.seed + 1)
        if args.model == "resnet18":
            print("synthetic mode: overriding --model resnet18 to --model tiny for fast local validation")
            args.model = "tiny"
    else:
        train_ds = CustomCIFAR100(root=args.data_root, train=True, download=True, transform=transform_train)
        valid_ds = CustomCIFAR100(root=args.data_root, train=False, download=True, transform=transform_test)

    forget_train_idx, retain_train_idx = split_indices(train_ds, args.forget_class, args, rng)
    forget_valid_idx, retain_valid_idx = split_indices(valid_ds, args.forget_class, args, rng)
    forget_train_idx = limit(forget_train_idx, args.max_forget_train, rng)
    retain_train_idx = limit(retain_train_idx, args.max_retain_train, rng)
    forget_valid_idx = limit(forget_valid_idx, args.max_forget_valid, rng)
    retain_valid_idx = limit(retain_valid_idx, args.max_retain_valid, rng)

    retain_subset_count = max(1, int(args.retain_subset_ratio * len(retain_train_idx)))
    retain_train_subset_idx = limit(retain_train_idx, retain_subset_count, rng)

    train_subset = ConcatDataset([Subset(train_ds, forget_train_idx), Subset(train_ds, retain_train_idx)])
    retain_train = Subset(train_ds, retain_train_idx)
    forget_train = Subset(train_ds, forget_train_idx)
    retain_train_subset = Subset(train_ds, retain_train_subset_idx)
    forget_valid = Subset(valid_ds, forget_valid_idx)
    retain_valid = Subset(valid_ds, retain_valid_idx)
    amnesiac_train = AmnesiacDataset(train_ds, forget_train_idx, retain_train_idx, seed=args.seed)

    loaders = {
        "train": make_loader(train_subset, args.batch_size, True, args.num_workers, pin_memory),
        "retain_train": make_loader(retain_train, args.batch_size, True, args.num_workers, pin_memory),
        "retain_train_subset": make_loader(retain_train_subset, args.batch_size, True, args.num_workers, pin_memory),
        "forget_train": make_loader(forget_train, args.batch_size, False, args.num_workers, pin_memory),
        "forget_valid": make_loader(forget_valid, args.batch_size, False, args.num_workers, pin_memory),
        "retain_valid": make_loader(retain_valid, args.batch_size, False, args.num_workers, pin_memory),
        "amnesiac_train": make_loader(amnesiac_train, args.batch_size, True, args.num_workers, pin_memory),
    }

    meta = {
        "device": device,
        "quick": args.quick,
        "forget_class": args.forget_class,
        "num_forget_train": len(forget_train_idx),
        "num_retain_train": len(retain_train_idx),
        "num_forget_valid": len(forget_valid_idx),
        "num_retain_valid": len(retain_valid_idx),
        "num_retain_train_subset": len(retain_train_subset_idx),
    }
    (out_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    stem = "TinyCNN" if args.model == "tiny" else "ResNET18"
    base_ckpt = out_dir / f"{stem}_CIFAR100Super20_ALL_CLASSES.pt"
    gold_ckpt = out_dir / f"{stem}_CIFAR100Super20_Gold_Class69.pt"

    def new_model(pretrained=None):
        if args.model == "tiny":
            return TinyCNN(num_classes=20).to(device)
        use_pretrained = args.pretrained if pretrained is None else pretrained
        return ResNet18(num_classes=20, pretrained=use_pretrained).to(device)

    def train_base():
        model = new_model()
        fit_one_cycle(args.base_epochs, model, loaders["train"], loaders["retain_valid"], device=device)
        torch.save(model.state_dict(), base_ckpt)

    load_or_train(base_ckpt, train_base, args.force_retrain)

    def load_base():
        model = new_model(pretrained=False)
        model.load_state_dict(torch.load(base_ckpt, map_location=device))
        return model

    rows = []
    started = time.time()
    incompetent_teacher = new_model(pretrained=False).eval()

    def full_eval(name, model):
        model.eval()
        forget_acc = evaluate(model, loaders["forget_valid"], device)["Acc"]
        retain_acc = evaluate(model, loaders["retain_valid"], device)["Acc"]
        mia = get_membership_attack_prob(
            loaders["retain_train_subset"], loaders["forget_train"], loaders["retain_valid"], model, device
        )
        zrf = compute_zrf(model, incompetent_teacher, loaders["forget_valid"], device)
        row = {
            "Method": name,
            "Forget_Acc": round(float(forget_acc), 2),
            "Retain_Acc": round(float(retain_acc), 2),
            "MIA": round(float(mia), 3),
            "ZRF": round(float(zrf), 3),
        }
        print(row)
        return row

    original = load_base()
    rows.append(full_eval("Original", original))

    if not args.skip_gold:
        def train_gold():
            gold = new_model()
            fit_one_cycle(args.gold_epochs, gold, loaders["retain_train"], loaders["retain_valid"], device=device)
            torch.save(gold.state_dict(), gold_ckpt)

        load_or_train(gold_ckpt, train_gold, args.force_retrain)
        gold = new_model(pretrained=False)
        gold.load_state_dict(torch.load(gold_ckpt, map_location=device))
        rows.append(full_eval("Gold", gold))

    amnesiac = load_base()
    fit_one_unlearning_cycle(
        args.amnesiac_epochs, amnesiac, loaders["amnesiac_train"], loaders["retain_valid"], lr=0.0001, device=device
    )
    rows.append(full_eval("Amnesiac", amnesiac))
    torch.save(amnesiac.state_dict(), out_dir / "amnesiac.pt")

    badteacher = load_base()
    blindspot_unlearner(
        model=badteacher,
        unlearning_teacher=incompetent_teacher,
        full_trained_teacher=load_base().eval(),
        retain_data=retain_train_subset,
        forget_data=forget_train,
        epochs=args.badteacher_epochs,
        lr=0.0001,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        KL_temperature=1,
    )
    rows.append(full_eval("Bad Teacher", badteacher))
    torch.save(badteacher.state_dict(), out_dir / "badteacher.pt")

    retain_aware = load_base()
    blindspot_unlearner_retain_aware(
        model=retain_aware,
        unlearning_teacher=incompetent_teacher,
        full_trained_teacher=load_base().eval(),
        retain_data=retain_train_subset,
        forget_data=forget_train,
        epochs=args.badteacher_epochs,
        lr=0.0001,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        KL_temperature=1,
        retain_ce_weight=args.retain_ce_weight,
    )
    rows.append(full_eval("Bad Teacher + Retain (Ours)", retain_aware))
    torch.save(retain_aware.state_dict(), out_dir / "retain_aware_badteacher.pt")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "unlearning_results.csv", index=False)
    plot_results(summary, out_dir)
    print(summary)
    print(f"elapsed_seconds={time.time() - started:.1f}")


if __name__ == "__main__":
    main()
