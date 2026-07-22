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
        quantisation: str | None = None,
    ):
        super().__init__()

        if quantisation is not None and quantisation not in ("float16", "int8", "int4", "binary"):
            raise ValueError(f"quantisation must be one of float16, int8, int4, binary, got {quantisation}")
        self.quantisation = quantisation

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
        self.logit_bias = nn.Parameter(torch.zeros(1))

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantisation is None:
            return x
        elif self.quantisation == "float16":
            return x.to(torch.float16)
        elif self.quantisation == "int8":
            abs_max = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
            x_q = (x / abs_max * 127).round().clamp(-128, 127)
            return x_q / 127 * abs_max
        elif self.quantisation == "int4":
            abs_max = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
            x_q = (x / abs_max * 7).round().clamp(-7, 7)
            return x_q / 7 * abs_max
        elif self.quantisation == "binary":
            return x.sign()
        raise ValueError(f"Unknown quantisation: {self.quantisation}")

    @classmethod
    def build_with_processor(
        cls,
        text_model_name: str = "prajjwal1/bert-tiny",
        vision_model_name: str = "facebook/dino-vits16",
        proj_dim: int = 256,
        quantisation: str | None = None,
        **kwargs,
    ):
        model = cls(
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            proj_dim=proj_dim,
            quantisation=quantisation,
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
        feats = self._quantize(feats)
        if normalize and self.quantisation != "binary":
            feats = F.normalize(feats, dim=-1)
        return feats

    def encode_image(self, pixel_values: torch.Tensor, normalize: bool = True):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        feats = outputs.last_hidden_state[:, 0, :]
        feats = self.vision_proj(feats)
        feats = self._quantize(feats)
        if normalize and self.quantisation != "binary":
            feats = F.normalize(feats, dim=-1)
        return feats

    def forward(self, pixel_values, input_ids, attention_mask):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds

    def clamp_logit_scale(self):
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.logit_scale_max)
