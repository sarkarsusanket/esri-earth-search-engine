"""
End-to-end trainer for VLM (BERT + DINO with contrastive learning).

Usage:
    python train.py \
        --parquet_path data/train.parquet \
        --batch_size 256 --epochs 10 --lr 1e-4 \
        --output_dir runs/vlm
"""
import argparse
import logging
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']="1,2"
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import build_dataloaders
from losses import (
    batch_retrieval_accuracy,
    binary_infonce_loss,
    binary_siglip_loss,
    clip_infonce_loss,
    siglip_loss,
)
from model import VLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train a VLM (TinyBERT+DINO) with contrastive learning.")
    # Data
    p.add_argument("--parquet_path", type=str, required=True)
    p.add_argument("--image_col", type=str, default="image")
    p.add_argument("--caption_col", type=str, default="caption")
    p.add_argument("--max_length", type=int, default=77)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--val_ratio", type=float, default=0.2)
    # Model
    p.add_argument("--text_model_name", type=str, default="prajjwal1/bert-tiny")
    p.add_argument("--vision_model_name", type=str, default="facebook/dino-vits16")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--quantisation", type=str, default=None,
                   choices=["float16", "int8", "int4", "binary"],
                   help="Quantize projected embeddings. binary uses Hamming distance.")
    p.add_argument("--siglip", action="store_true",
                   help="Use SigLIP sigmoid loss instead of InfoNCE")
    # Optimization
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    # Bookkeeping
    p.add_argument("--output_dir", type=str, default="runs/vlm")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, siglip: bool = False):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        image_embeds, text_embeds = model(pixel_values, input_ids, attention_mask)

        if siglip:
            if model.quantisation == "binary":
                loss, logits = binary_siglip_loss(
                    image_embeds, text_embeds, model.logit_scale.exp(), model.logit_bias
                )
            else:
                loss, logits = siglip_loss(
                    image_embeds, text_embeds, model.logit_scale.exp(), model.logit_bias
                )
        else:
            if model.quantisation == "binary":
                loss, logits = binary_infonce_loss(
                    image_embeds, text_embeds, model.logit_scale.exp()
                )
            else:
                loss, logits = clip_infonce_loss(
                    image_embeds, text_embeds, model.logit_scale.exp()
                )

        total_loss += loss.item()
        total_acc += batch_retrieval_accuracy(logits)
        n_batches += 1

    model.train()
    if n_batches == 0:
        return float("nan"), float("nan")
    return total_loss / n_batches, total_acc / n_batches


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, args):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "args": vars(args),
        },
        path,
    )
    logger.warning("Saved checkpoint: %s", path)


def main():
    args = parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    logger.warning("Using device: %s", device)

    model, processor, tokenizer = VLM.build_with_processor(
        text_model_name=args.text_model_name,
        vision_model_name=args.vision_model_name,
        proj_dim=args.proj_dim,
        quantisation=args.quantisation,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.warning(
        "Model params: %d total, %d trainable (%.2f%%)",
        total_params,
        trainable_params,
        100 * trainable_params / total_params if total_params > 0 else 0,
    )

    train_dataset, train_loader, val_loader = build_dataloaders(
        args.parquet_path,
        processor,
        tokenizer,
        image_col=args.image_col,
        caption_col=args.caption_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    logger.warning(
        "Train examples: %d | Val examples: %d | steps/epoch: %d | total steps: %d",
        len(train_dataset),
        len(val_loader.dataset),
        steps_per_epoch,
        total_steps,
    )

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "logit_scale" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch, global_step = 0, 0
    if args.resume_from:
        logger.warning("Resuming from %s", args.resume_from)
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]

    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        running_loss, running_acc = 0.0, 0.0
        epoch_loss, epoch_acc, epoch_batches = 0.0, 0.0, 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=args.amp,
            ):
                image_embeds, text_embeds = model(
                    pixel_values, input_ids, attention_mask
                )
                if args.siglip:
                    if model.quantisation == "binary":
                        loss, logits = binary_siglip_loss(
                            image_embeds, text_embeds, model.logit_scale.exp(), model.logit_bias
                        )
                    else:
                        loss, logits = siglip_loss(
                            image_embeds, text_embeds, model.logit_scale.exp(), model.logit_bias
                        )
                else:
                    if model.quantisation == "binary":
                        loss, logits = binary_infonce_loss(
                            image_embeds, text_embeds, model.logit_scale.exp()
                        )
                    else:
                        loss, logits = clip_infonce_loss(
                            image_embeds, text_embeds, model.logit_scale.exp()
                        )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.clamp_logit_scale()

            global_step += 1
            running_loss += loss.item()
            running_acc += batch_retrieval_accuracy(logits)
            epoch_loss += loss.item()
            epoch_acc += batch_retrieval_accuracy(logits)
            epoch_batches += 1

            if global_step % args.log_every == 0:
                if args.verbose >= 1:
                    avg_loss = running_loss / args.log_every
                    avg_acc = running_acc / args.log_every
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        "epoch %d | step %d/%d | loss %.4f | in-batch acc %.3f | "
                        "lr %.2e | temp %.2f",
                        epoch,
                        global_step,
                        total_steps,
                        avg_loss,
                        avg_acc,
                        lr_now,
                        model.logit_scale.exp().item(),
                    )
                running_loss, running_acc = 0.0, 0.0

        logger.warning(
            "Epoch %d finished in %.1fs", epoch, time.time() - epoch_start
        )

        val_loss, val_acc = evaluate(model, val_loader, device, siglip=args.siglip)
        logger.warning(
            "Epoch %d | val_loss %.4f | val_acc %.3f", epoch, val_loss, val_acc
        )
        current_loss = val_loss

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pt")
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler, epoch + 1, global_step, args
            )
        else:
            patience_counter += 1
            logger.warning(
                "EarlyStopping: no improvement for %d/%d epochs",
                patience_counter,
                args.patience,
            )
            if patience_counter >= args.patience:
                logger.warning(
                    "Early stopping triggered after %d epochs", epoch
                )
                break

    logger.warning("Training complete.")


if __name__ == "__main__":
    main()
