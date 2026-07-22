"""Symmetric InfoNCE (CLIP-style contrastive) loss."""
import torch
import torch.nn.functional as F


def clip_infonce_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
):
    """
    image_embeds, text_embeds: (B, D), assumed L2-normalized.
    logit_scale: scalar log-temperature parameter (already exponentiated
        by the caller, or pass logit_scale.exp() — see train.py).

    Returns (loss, logits_per_image) for optional accuracy logging.
    """
    device = image_embeds.device
    batch_size = image_embeds.shape[0]

    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(batch_size, device=device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i2t + loss_t2i) / 2

    return loss, logits_per_image


@torch.no_grad()
def batch_retrieval_accuracy(logits_per_image: torch.Tensor) -> float:
    """Fraction of images whose top-1 nearest text (within the batch) is
    the paired caption. Cheap in-batch sanity metric, not a real eval."""
    batch_size = logits_per_image.shape[0]
    preds = logits_per_image.argmax(dim=-1)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    return (preds == labels).float().mean().item()


def binary_infonce_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
):
    """
    image_embeds, text_embeds: (B, D), assumed {-1, +1} binary.
    Uses Hamming distance as the similarity metric.
    """
    device = image_embeds.device
    batch_size = image_embeds.shape[0]
    D = image_embeds.shape[-1]

    hamming = 0.5 * (D - image_embeds @ text_embeds.t())
    logits_per_image = -hamming * logit_scale
    logits_per_text = logits_per_image.t()

    labels = torch.arange(batch_size, device=device)
    loss = (F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)) / 2

    return loss, logits_per_image


def siglip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
):
    """
    SigLIP pairwise sigmoid loss (https://arxiv.org/abs/2303.15343).
    image_embeds, text_embeds: (B, D), assumed L2-normalized.
    """
    device = image_embeds.device
    batch_size = image_embeds.shape[0]

    logits = logit_scale * image_embeds @ text_embeds.t() + logit_bias
    labels = 2 * torch.eye(batch_size, device=device) - 1
    loss = -F.logsigmoid(labels * logits).sum() / batch_size

    return loss, logits


def binary_siglip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
):
    """
    SigLIP loss with binary embeddings and Hamming distance.
    image_embeds, text_embeds: (B, D), assumed {-1, +1} binary.
    """
    device = image_embeds.device
    batch_size = image_embeds.shape[0]
    D = image_embeds.shape[-1]

    hamming = 0.5 * (D - image_embeds @ text_embeds.t())
    logits = -hamming * logit_scale + logit_bias
    labels = 2 * torch.eye(batch_size, device=device) - 1
    loss = -F.logsigmoid(labels * logits).sum() / batch_size

    return loss, logits
