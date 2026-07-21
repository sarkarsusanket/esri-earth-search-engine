# VampireCLIP

Minimal end-to-end trainer for a CLIP-style dual encoder, initialized from
pretrained CLIP weights, with an extra linear projection head appended to
both the image and text towers to bring embeddings down to a chosen `D`.

## Layout

```
esri_rs_clip/
  data.py     # parquet (image bytes + caption) -> DataLoader
  model.py    # pretrained CLIP backbone + image_proj / text_proj -> D dims
  losses.py   # symmetric InfoNCE loss
  train.py    # training loop: AMP, cosine warmup schedule, checkpointing
```

## Data format

A parquet file with:
- an image column: raw image bytes (`bytes`), or a struct `{"bytes": ...}`
  (the format HF `datasets`' `Image` feature writes to parquet)
- a caption column: `string`

## Model

`VampireCLIP` wraps `transformers.CLIPModel` (initialized from a pretrained
checkpoint, e.g. `openai/clip-vit-base-patch32`) and adds:

```
image: pixel_values -> CLIP vision tower -> CLIP's own projection (clip_dim)
       -> image_proj (Linear, no bias) -> D-dim, L2-normalized

text:  input_ids -> CLIP text tower -> CLIP's own projection (clip_dim)
       -> text_proj (Linear, no bias) -> D-dim, L2-normalized
```

`image_proj` / `text_proj` are randomly initialized and trained from
scratch; the rest of the backbone starts from pretrained weights (and can
optionally be frozen with `--freeze_backbone`).

## Loss

Standard symmetric InfoNCE / CLIP loss: cross-entropy over the in-batch
image-to-text and text-to-image similarity matrices, scaled by a learned
temperature (`logit_scale`, clamped to `log(100)` as in the original CLIP
paper).

## Usage

```bash
pip install -r requirements.txt

python train.py --parquet_path E:\Data\query-earth\vlm_captions.parquet --val_parquet_path E:\Data\query-earth\vlm_captions_2.parquet --image_col image_bytes --caption_col caption --clip_name openai/clip-vit-base-patch32 --proj_dim 128 --batch_size 256 --epochs 100  --lr 1e-4 --output_dir D:\Code\query-earth\vlm\runs
```

Resume training:

```bash
python train.py ... --resume_from runs/vampire_clip/checkpoint_epoch3.pt
```

## Notes / things to tune for a real run

- `--batch_size`: InfoNCE quality scales with in-batch negatives, so bigger
  is generally better if memory allows. Consider gradient accumulation or
  a memory-queue (MoCo-style) for very large effective batch sizes on
  limited hardware — not implemented here.
- `--freeze_backbone`: useful for a quick sanity check that only the new
  projection heads can learn something before committing to full
  fine-tuning.
- Swap `CLIPModel`/`CLIPProcessor` for `open_clip` if you want access to
  larger/more open pretrained checkpoints (e.g. LAION ones) — `model.py`
  is the only file that would need to change.
