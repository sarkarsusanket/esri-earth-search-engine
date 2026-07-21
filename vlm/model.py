"""
VampireCLIP: a CLIP model initialized from pretrained weights, with an
additional linear projection layer bolted onto the end of both the image
and text encoders to bring the embedding dimension down to `proj_dim`.

The base CLIP image/text towers (including their own built-in projections)
are kept and initialized from pretrained weights; the extra `image_proj` /
`text_proj` linear layers are trained from scratch (or fine-tuned) on top.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

logger = logging.getLogger(__name__)


class VampireCLIP(nn.Module):
    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        proj_dim: int = 256,
        freeze_backbone: bool = False,
        logit_scale_init: float = 1 / 0.07,
        logit_scale_max: float = 100.0,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        clip_dim = self.clip.config.projection_dim

        self.image_proj = nn.Linear(clip_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(clip_dim, proj_dim, bias=False)
        nn.init.normal_(self.image_proj.weight, std=clip_dim ** -0.5)
        nn.init.normal_(self.text_proj.weight, std=clip_dim ** -0.5)

        # Learned temperature, same convention as the original CLIP paper.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.logit_scale_max = math.log(logit_scale_max)

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad_(False)
            logger.info("CLIP backbone frozen; only training projection heads.")

    @classmethod
    def build_with_processor(
        cls,
        clip_name: str = "openai/clip-vit-base-patch32",
        proj_dim: int = 256,
        **kwargs,
    ):
        """Convenience constructor that also returns the matching
        processor (image transforms) and tokenizer for `clip_name`."""
        model = cls(clip_name=clip_name, proj_dim=proj_dim, **kwargs)
        processor = CLIPProcessor.from_pretrained(clip_name).image_processor
        tokenizer = CLIPTokenizerFast.from_pretrained(clip_name)
        return model, processor, tokenizer

    def encode_image(self, pixel_values: torch.Tensor, normalize: bool = True):
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        feats = self.image_proj(outputs)
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ):
        outputs = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        feats = self.text_proj(outputs)

        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds

    def clamp_logit_scale(self):
        """Call after each optimizer step, matching the original CLIP
        implementation, which caps temperature at 100."""
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.logit_scale_max)
