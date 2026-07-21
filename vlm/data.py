"""
Dataset utilities for training a CLIP-style model from a parquet file
containing raw image bytes and text captions.
"""
import io
import logging

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Some corrupt/truncated images shouldn't crash a multi-hour training run.
Image.MAX_IMAGE_PIXELS = None


class ParquetImageTextDataset(Dataset):
    """Reads a parquet file with an image-bytes column and a caption column.

    The parquet is loaded fully into memory as a pandas DataFrame (only the
    two needed columns), which is fine up to a few million rows. For larger
    datasets, swap this out for a pyarrow.dataset streaming reader.
    """

    def __init__(
        self,
        parquet_path: str,
        image_col: str = "image",
        caption_col: str = "caption",
    ):
        self.image_col = image_col
        self.caption_col = caption_col

        logger.info("Loading parquet: %s", parquet_path)
        self.df = pd.read_parquet(parquet_path, columns=[image_col, caption_col])
        self.df = self.df.dropna(subset=[image_col, caption_col]).reset_index(drop=True)
        logger.info("Loaded %d rows", len(self.df))

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_bytes = row[self.image_col]
        # Parquet may store bytes directly, or as a dict like {"bytes": ...}
        # (common when exported from HF datasets' Image feature).
        if isinstance(img_bytes, dict) and "bytes" in img_bytes:
            img_bytes = img_bytes["bytes"]

        try:
            image = self._load_image(img_bytes)
        except Exception as e:  # noqa: BLE001 - skip bad rows, don't crash training
            logger.warning("Failed to decode image at row %d: %s", idx, e)
            return self.__getitem__((idx + 1) % len(self))

        caption = str(row[self.caption_col])
        return image, caption


class Collator:
    """Turns a list of (PIL.Image, caption) into model-ready tensors.

    Wrapped in a class (rather than a closure) so it's picklable for
    multi-worker DataLoaders.
    """

    def __init__(self, processor, tokenizer, max_length: int = 77):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images, captions = zip(*batch)

        pixel_values = self.processor(images=list(images), return_tensors="pt")[
            "pixel_values"
        ]

        text_inputs = self.tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }


def build_dataloader(
    parquet_path: str,
    processor,
    tokenizer,
    image_col: str = "image",
    caption_col: str = "caption",
    batch_size: int = 256,
    num_workers: int = 0,
    max_length: int = 77,
    shuffle: bool = True,
    drop_last: bool = True,
):
    dataset = ParquetImageTextDataset(parquet_path, image_col, caption_col)
    collator = Collator(processor, tokenizer, max_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader
