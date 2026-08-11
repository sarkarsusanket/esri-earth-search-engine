"""
Zero-shot top-1 accuracy benchmark for a trained VLM checkpoint on
standard remote-sensing scene classification datasets:

    OPTIMAL31, RSC11, RSICB128, WHU-RS19, RSSCN7 (aka RS2800/RSSCN7), CLRS

All six are assumed to be laid out on disk as a standard "ImageFolder":

    <dataset_root>/
        class_name_1/
            img001.jpg
            ...
        class_name_2/
            ...

This is how each of these datasets is normally distributed/unzipped, so no
per-dataset special-casing is needed beyond pointing at the right root and
(optionally) supplying a class-name-cleanup map for datasets whose folder
names are abbreviated (e.g. "arboretum" vs "Arboretum") or use different
capitalization/underscore conventions than the token you want fed into the
text encoder.

For each dataset we build a zero-shot classifier by embedding the class
names with a set of remote-sensing-flavored prompt templates (averaged, as
in the original CLIP paper), then classify every image by nearest cosine
similarity to the per-class text embedding.

Usage:
    python eval.py --checkpoint D:\Code\query-earth\vlm\runs\checkpoint_epoch9.pt --dataset_config D:\Code\query-earth\vlm\eval\dataset_paths.json --batch_size 256

Where dataset_paths.json looks like:
    {
        "OPTIMAL31": "/data/rs_benchmarks/OPTIMAL-31",
        "RSC11": "/data/rs_benchmarks/RSC11",
        "RSICB128": "/data/rs_benchmarks/RSI-CB128",
        "WHU-RS19": "/data/rs_benchmarks/WHU-RS19",
        "RSSCN7": "/data/rs_benchmarks/RSSCN7",
        "CLRS": "/data/rs_benchmarks/CLRS"
    }

Datasets can also be pointed to individually on the command line, e.g.
    --root OPTIMAL31=/data/rs_benchmarks/OPTIMAL-31
(repeatable; overrides/extends anything in --dataset_config).
"""
import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(rf"D:\Code\query-earth\vlm")
from model import VLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Canonical order the user asked for. "RSSCN7" covers the dataset commonly
# labeled "RS2800/RSSCN7" (2800 images, 7 classes).
DATASET_KEYS = ["Optimal31", "RSC11", "RSICB128"]

# Prompt templates adapted from CLIP's zero-shot template set for
# aerial/satellite imagery, as used in remote-sensing CLIP zero-shot papers.
RS_TEMPLATES = [
    "{}.",
]


def clean_classname(name: str) -> str:
    """Folder names -> readable phrase, e.g. 'dense_residential' ->
    'dense residential'. Datasets are free to override individual names
    via a JSON class-name map (see --classname_map)."""
    name = name.replace("_", " ").replace("-", " ").strip()
    # crude article choice; "an" before vowel sounds
    return name.lower()


class ZeroShotFolderDataset(ImageFolder):
    """torchvision ImageFolder that hands back raw PIL images (transform
    is applied later, in the collate function, via the CLIP processor) so
    preprocessing exactly matches training."""

    def __init__(self, root):
        super().__init__(root, transform=None, loader=self._safe_loader)

    @staticmethod
    def _safe_loader(path):
        from PIL import Image

        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class EvalCollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, labels = zip(*batch)
        pixel_values = self.processor(images=list(images), return_tensors="pt")[
            "pixel_values"
        ]
        labels = torch.tensor(labels, dtype=torch.long)
        return pixel_values, labels


@torch.no_grad()
def build_zero_shot_classifier(
    model: VLM,
    tokenizer,
    classnames: List[str],
    templates: List[str],
    device,
) -> torch.Tensor:
    """Returns a (D, num_classes) matrix of L2-normalized, template-averaged
    text embeddings — one column per class."""
    model.eval()
    weights = []
    for name in classnames:
        phrase = clean_classname(name)
        prompts = [t.format(phrase) for t in templates]

        tok = tokenizer(
            prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        text_embeds = model.encode_text(
            tok["input_ids"], tok["attention_mask"], normalize=True
        )  # (num_templates, D)
        class_embed = text_embeds.mean(dim=0)
        class_embed = F.normalize(class_embed, dim=-1)
        weights.append(class_embed)

    return torch.stack(weights, dim=1)  # (D, num_classes)


@torch.no_grad()
def evaluate_dataset(
    model: VLM,
    processor,
    tokenizer,
    root: str,
    device,
    batch_size: int = 256,
    num_workers: int = 8,
    templates: Optional[List[str]] = None,
) -> Dict:
    templates = templates or RS_TEMPLATES

    dataset = ZeroShotFolderDataset(root)
    classnames = dataset.classes  # folder names, index-aligned with targets
    logger.info(
        "Loaded %s: %d images, %d classes", root, len(dataset), len(classnames)
    )

    classifier_weights = build_zero_shot_classifier(
        model, tokenizer, classnames, templates, device
    )  # (D, num_classes)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=EvalCollate(processor),
        pin_memory=True,
    )

    correct, top5_correct, total = 0, 0, 0
    model.eval()
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        image_embeds = model.encode_image(pixel_values, normalize=True)  # (B, D)
        logits = image_embeds @ classifier_weights  # (B, num_classes)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()

        k = min(5, logits.shape[-1])
        top5 = logits.topk(k, dim=-1).indices
        top5_correct += (top5 == labels.unsqueeze(1)).any(dim=-1).sum().item()

        total += labels.shape[0]

    return {
        "num_images": total,
        "num_classes": len(classnames),
        "top1_acc": correct / total,
        "top5_acc": top5_correct / total,
    }


def load_model_from_checkpoint(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt["args"]

    model, processor, tokenizer = VLM.build_with_processor(
        text_model_name=train_args.get("text_model_name", "prajjwal1/bert-tiny"),
        vision_model_name=train_args.get("vision_model_name", "facebook/dino-vits16"),
        proj_dim=train_args["proj_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint %s (epoch %s, step %s)",
        checkpoint_path,
        ckpt.get("epoch"),
        ckpt.get("step"),
    )
    return model, processor, tokenizer


def resolve_dataset_roots(args) -> Dict[str, str]:
    roots: Dict[str, str] = {}
    if args.dataset_config:
        with open(args.dataset_config, "r") as f:
            roots.update(json.load(f))

    for entry in args.root or []:
        if "=" not in entry:
            raise ValueError(f"--root entries must be name=path, got: {entry}")
        name, path = entry.split("=", 1)
        roots[name] = path

    missing = [k for k in DATASET_KEYS if k not in roots]
    if missing:
        logger.warning(
            "No path provided for: %s (skipping). Supply via --dataset_config "
            "or --root NAME=path.",
            ", ".join(missing),
        )
    unknown = [k for k in roots if k not in DATASET_KEYS]
    if unknown:
        logger.warning(
            "Paths given for unrecognized dataset keys (evaluating anyway): %s",
            ", ".join(unknown),
        )

    return roots


def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot top-1 eval on remote-sensing benchmarks."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument(
        "--root",
        action="append",
        help="Override/add a dataset root: NAME=/path/to/dataset. Repeatable.",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model, processor, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
    dataset_roots = resolve_dataset_roots(args)

    results = {}
    for name in DATASET_KEYS:
        if name not in dataset_roots:
            continue
        root = dataset_roots[name]
        if not os.path.isdir(root):
            logger.warning("Skipping %s: path does not exist: %s", name, root)
            continue
        try:
            results[name] = evaluate_dataset(
                model,
                processor,
                tokenizer,
                root,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed evaluating %s: %s", name, e)

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"{'Dataset':<12} {'#images':>8} {'#classes':>9} {'Top-1':>8} {'Top-5':>8}")
    print("-" * 60)
    top1s = []
    for name in DATASET_KEYS:
        if name not in results:
            print(f"{name:<12} {'--':>8} {'--':>9} {'--':>8} {'--':>8}")
            continue
        r = results[name]
        print(
            f"{name:<12} {r['num_images']:>8} {r['num_classes']:>9} "
            f"{r['top1_acc']*100:>7.2f}% {r['top5_acc']*100:>7.2f}%"
        )
        top1s.append(r["top1_acc"])
    print("-" * 60)
    if top1s:
        print(f"{'Mean top-1':<12} {'':>8} {'':>9} {sum(top1s)/len(top1s)*100:>7.2f}%")
    print("=" * 60 + "\n")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote results to %s", args.output_json)


if __name__ == "__main__":
    main()