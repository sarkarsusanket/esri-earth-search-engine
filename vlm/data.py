import io
import logging

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


class ParquetImageTextDataset(Dataset):
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
        if isinstance(img_bytes, dict) and "bytes" in img_bytes:
            img_bytes = img_bytes["bytes"]

        try:
            image = self._load_image(img_bytes)
        except Exception as e:
            logger.warning("Failed to decode image at row %d: %s", idx, e)
            return self.__getitem__((idx + 1) % len(self))

        caption = str(row[self.caption_col])
        return image, caption


class Collator:
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


def split_dataset(dataset, val_ratio: float = 0.2, seed: int = 42):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def build_dataloaders(
    parquet_path: str,
    processor,
    tokenizer,
    image_col: str = "image",
    caption_col: str = "caption",
    batch_size: int = 256,
    num_workers: int = 0,
    max_length: int = 77,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    dataset = ParquetImageTextDataset(parquet_path, image_col, caption_col)
    collator = Collator(processor, tokenizer, max_length)

    train_dataset, val_dataset = split_dataset(dataset, val_ratio, seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        collate_fn=collator,
        drop_last=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    logger.info(
        "Split %d examples into train=%d val=%d (%.1f/%.1f)",
        len(dataset),
        len(train_dataset),
        len(val_dataset),
        (1 - val_ratio) * 100,
        val_ratio * 100,
    )

    return train_dataset, train_loader, val_loader
