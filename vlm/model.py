import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoImageProcessor

logger = logging.getLogger(__name__)


class VLM(nn.Module):
    def __init__(
        self,
        text_model_name: str = "prajjwal1/bert-tiny",
        vision_model_name: str = "facebook/dino-vits16",
        proj_dim: int = 256,
        logit_scale_init: float = 1 / 0.07,
        logit_scale_max: float = 100.0,
    ):
        super().__init__()

        self.text_encoder = BertModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        vision_dim = self.vision_encoder.config.hidden_size
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)

        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.vision_proj = nn.Linear(vision_dim, proj_dim, bias=False)
        nn.init.normal_(self.text_proj.weight, std=text_dim ** -0.5)
        nn.init.normal_(self.vision_proj.weight, std=vision_dim ** -0.5)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.logit_scale_max = math.log(logit_scale_max)

    @classmethod
    def build_with_processor(
        cls,
        text_model_name: str = "prajjwal1/bert-tiny",
        vision_model_name: str = "facebook/dino-vits16",
        proj_dim: int = 256,
        **kwargs,
    ):
        model = cls(
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            proj_dim=proj_dim,
            **kwargs,
        )
        tokenizer = BertTokenizer.from_pretrained(text_model_name)
        processor = AutoImageProcessor.from_pretrained(vision_model_name)
        return model, processor, tokenizer

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        feats = outputs.pooler_output
        feats = self.text_proj(feats)
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    def encode_image(self, pixel_values: torch.Tensor, normalize: bool = True):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        feats = outputs.last_hidden_state[:, 0, :]
        feats = self.vision_proj(feats)
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds

    def clamp_logit_scale(self):
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.logit_scale_max)
