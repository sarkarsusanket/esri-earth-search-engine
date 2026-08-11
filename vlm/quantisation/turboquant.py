import os
import numpy as np
import torch

def quantize_and_save_7m(
    input_npz_path: str,
    output_dir: str,
    num_chunks: int = 4,
    batch_size: int = 500_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Memory-map input NPZ to avoid loading 14GB into RAM at once
    logger_print("Opening input NPZ file...")
    npz = np.load(input_npz_path, mmap_mode="r")
    
    # Expecting keys 'emb' and 'centers'
    raw_embeds = npz["emb"]      # shape (N, D)
    raw_centers = npz["centers"]  # shape (N, 2)
    
    N, D = raw_embeds.shape
    chunk_dim = D // num_chunks
    assert D % num_chunks == 0, f"Dimension {D} must be divisible by num_chunks {num_chunks}"

    print(f"Dataset size: {N:,} vectors | Dimension: {D} | Chunks: {num_chunks}")
    print(f"Using device: {device}")

    # 2. Generate Orthogonal Rotation Matrix (D x D) and save immediately
    q_mat, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    q_mat_cpu = q_mat.cpu().numpy()
    np.save(os.path.join(output_dir, "rotation_matrix.npy"), q_mat_cpu)

    # 3. Pre-allocate disk output files via memmap
    codes_path = os.path.join(output_dir, "quantized_codes.npy")
    scales_path = os.path.join(output_dir, "scales.npy")
    mins_path = os.path.join(output_dir, "mins.npy")
    centers_path = os.path.join(output_dir, "centers.npy")

    codes_mm = np.lib.format.open_memmap(codes_path, mode="w+", dtype=np.uint8, shape=(N, D))
    scales_mm = np.lib.format.open_memmap(scales_path, mode="w+", dtype=np.float32, shape=(N, num_chunks))
    mins_mm = np.lib.format.open_memmap(mins_path, mode="w+", dtype=np.float32, shape=(N, num_chunks))
    centers_mm = np.lib.format.open_memmap(centers_path, mode="w+", dtype=raw_centers.dtype, shape=(N, 2))

    # 4. Stream & Quantize in Batches
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        print(f"Processing batch {start_idx:,} to {end_idx:,}...")

        # Load batch to GPU
        batch_emb = torch.from_numpy(raw_embeds[start_idx:end_idx]).to(device, dtype=torch.float32)
        batch_centers = raw_centers[start_idx:end_idx]

        # Step A: Rotate batch (Polar Transformation)
        rotated = batch_emb @ q_mat  # (batch_len, D)

        # Step B: Reshape into chunks
        batch_len = rotated.shape[0]
        chunks = rotated.view(batch_len, num_chunks, chunk_dim)

        # Step C: Compute min/max & scales per chunk
        c_min = chunks.min(dim=-1, keepdim=True).values
        c_max = chunks.max(dim=-1, keepdim=True).values
        c_scale = torch.clamp((c_max - c_min) / 255.0, min=1e-8)

        # Step D: Quantize to uint8
        quantized = torch.round((chunks - c_min) / c_scale).to(torch.uint8)
        quantized = quantized.view(batch_len, D)  # Flatten back to (batch_len, D)

        # Step E: Stream directly to disk via memmap
        codes_mm[start_idx:end_idx] = quantized.cpu().numpy()
        scales_mm[start_idx:end_idx] = c_scale.squeeze(-1).cpu().numpy()
        mins_mm[start_idx:end_idx] = c_min.squeeze(-1).cpu().numpy()
        centers_mm[start_idx:end_idx] = batch_centers

        # Flush disk buffer
        codes_mm.flush()
        scales_mm.flush()
        mins_mm.flush()
        centers_mm.flush()

    print(f"\nSuccessfully compressed and saved 7M vectors to '{output_dir}'!")

def logger_print(msg):
    print(f"[TurboQuant] {msg}")

# Run Example
if __name__ == "__main__":
    quantize_and_save_7m(
        input_npz_path=rf"E:\Data\query-earth\embeddings_california\hex7-skyclip-low.npz",
        output_dir=rf"E:\Data\query-earth\embeddings_california\vision-quantised-lowres",
        num_chunks=4,
        batch_size=100_000
    )