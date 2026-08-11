import io
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

# =====================================================================
# Configuration
# =====================================================================
PARQUET_FILE_PATH = rf"E:\Data\query-earth\vlm_captions_2.parquet"
IMAGE_COLUMN_NAME = "image_bytes"

CLIP_VISION_MODEL_NAME = "ViT-L-14"
CLIP_VISION_PRETRAINED = "laion2b_s32b_b82k"

# Sampling and Batching
SAMPLE_SIZE = 100000   # Set N images to sample (or None to use all)
BATCH_SIZE = 128     # Image processing batch size for VRAM optimization
RANDOM_SEED = 42

QUERIES = [
    # --- Urban & Residential Infrastructure (1-20) ---
    "an aerial view of a dense residential neighborhood with red tile roofs",
    "a top-down satellite shot of a suburban cul-de-sac with parked cars",
    "overhead drone shot of a high-rise city center with glass skyscrapers",
    "a high-altitude aerial view of an urban grid layout at night",
    "bird's-eye view of a construction site with heavy excavators and cranes",
    "satellite image of a coastal city with a large shipping port",
    "an aerial view of a sprawling industrial park with large warehouse roofs",
    "aerial view of a luxury resort with swimming pools near the beach",
    "bird's-eye view of a university campus with sports fields and green courtyards",
    "top-down view of a historical European old town with narrow cobblestone streets",
    "an aerial view of informal settlements or favelas packed tightly on a hillside",
    "a satellite image of a circular housing development layout",
    "overhead view of solar panels installed across industrial factory rooftops",
    "bird's-eye view of a parking lot full of multi-colored automobiles",
    "top-down view of a sewage treatment plant with round settling tanks",
    "an aerial view of a water park with colorful slides and winding pools",
    "a high-altitude view of a stadium surrounded by surface parking lots",
    "satellite perspective of a mobile home park with uniform structures",
    "overhead drone view of a cemetery with neatly aligned gravestones",
    "an aerial view of a power station with cooling towers emitting steam",

    # --- Transport & Logistics Infrastructure (21-40) ---
    "top-down aerial view of a complex highway interchange or cloverleaf junction",
    "bird's-eye view of a major international airport with airplanes parked at gates",
    "a satellite shot of a cargo container terminal filled with colorful shipping containers",
    "overhead view of a long freight train moving across rural railway tracks",
    "an aerial perspective of a suspension bridge spanning a wide river",
    "top-down drone view of a multi-lane toll booth plaza with heavy traffic",
    "a high-altitude view of an airplane runway adjacent to a grassy airfield",
    "bird's-eye view of a marina packed with white sailboats and luxury yachts",
    "an aerial shot of a roundabout junction with surrounding green space",
    "satellite image of a dry dock facility holding a large naval vessel",
    "overhead perspective of a busy canal with cargo barges navigating through",
    "an aerial view of a train switching yard with dozens of parallel tracks",
    "top-down view of a pier extending out into deep blue water",
    "bird's-eye view of an elevated highway cutting through a dense downtown district",
    "overhead view of a remote asphalt airstrip in the desert",
    "an aerial view of a bus terminal with parked public transportation vehicles",
    "bird's-eye view of a footbridge crossing over a busy multi-lane highway",
    "satellite image of a ferry terminal with cars queuing to board",
    "top-down aerial shot of a curved mountain road with hairpin turns",
    "an aerial view of a cruise ship docked at an ocean port",

    # --- Agriculture, Forestry & Land Use (41-60) ---
    "top-down satellite image of center-pivot irrigation crops creating green circles",
    "an aerial view of geometric farmland fields with varying crop colors",
    "bird's-eye view of a dense green forest canopy with a winding road cutting through",
    "a satellite view of terraced rice fields built into a steep hillside",
    "overhead drone view of neatly planted rows in a vineyard",
    "an aerial view of a large cattle feedlot with livestock pens",
    "top-down perspective of an orchard with structured rows of fruit trees",
    "bird's-eye view of a greenhouse complex with reflective glass roofs",
    "an aerial view of freshly plowed brown soil adjacent to green pasture",
    "satellite shot of deforestation boundaries showing clear-cut forest patches",
    "overhead view of a timber logging site with felled trees gathered together",
    "an aerial view of a wind farm with large white turbines scattered over farmland",
    "top-down view of a fish farm with floating circular aquaculture pens",
    "bird's-eye view of salt evaporation ponds with vivid pink and orange hues",
    "an aerial view of a farm homestead with barns, silos, and a farmhouse",
    "overhead drone perspective of a sunflower field in bloom",
    "satellite view of an oil palm plantation with symmetrical tree spacing",
    "an aerial view of a grain elevator facility next to rail lines",
    "top-down shot of a dried-up agricultural field showing soil cracks",
    "bird's-eye view of a commercial turf farm with bright green grass strips",

    # --- Natural Landscapes & Water Bodies (61-80) ---
    "top-down aerial view of a meandering river carving through a canyon",
    "a satellite view of a coral reef under shallow turquoise sea water",
    "bird's-eye view of ocean waves breaking against a rocky coastline",
    "an aerial view of a glacier flowing between snow-capped mountain peaks",
    "top-down perspective of a delta network branching into a body of water",
    "satellite image of an active volcano crater surrounded by dark lava fields",
    "an aerial view of sand dunes in a desert forming wave-like patterns",
    "bird's-eye view of a alpine lake with crystal clear turquoise water",
    "overhead drone view of a lush mangrove swamp with intricate waterways",
    "an aerial view of a dense tropical rainforest canopy",
    "top-down view of a salt flat expanse with white dry crust",
    "satellite perspective of an island surrounded by open deep ocean",
    "an aerial view of a wetland marsh with patches of reeds and standing water",
    "bird's-eye view of a canyon floor with a narrow river flowing through it",
    "overhead view of a snow-covered forest with pine trees",
    "an aerial perspective of a barrier island separating ocean from lagoon",
    "top-down shot of a waterfall cascading off a steep cliff side",
    "satellite view of a dry lake bed or playa in an arid region",
    "bird's-eye view of an ocean bay filled with small coastal islands",
    "an aerial view of geothermal hot springs with bright mineral colors",

    # --- Industry, Mining & Energy (81-100) ---
    "top-down satellite image of a massive open-pit mine with stepped terraces",
    "an aerial view of an oil refinery with storage tanks, pipes, and flare stacks",
    "bird's-eye view of a solar power plant with thousands of photovoltaic panels",
    "overhead drone view of a hydroelectric dam holding back a large reservoir",
    "an aerial view of a quarry with gravel stockpiles and heavy machinery",
    "satellite image of cylindrical white oil storage tanks arranged in a grid",
    "top-down view of an offshore oil platform standing in open water",
    "an aerial view of a coal mine with dark piles and conveyor belts",
    "bird's-eye view of a wind turbine farm constructed along a coastal ridge",
    "overhead view of a scrap yard full of crushed metal and junk cars",
    "an aerial perspective of a geothermal energy power plant in a volcanic area",
    "top-down shot of an industrial retention pond filled with colored wastewater",
    "satellite view of a natural gas terminal with spherical gas tanks",
    "an aerial view of an active logging yard with timber stacked high",
    "bird's-eye view of an offshore wind farm surrounded by ocean water",
    "top-down view of a water reservoir retaining wall and spillway",
    "an aerial view of a concrete mixing plant with sand and gravel mounds",
    "satellite image of an underground mining surface facility",
    "overhead drone view of a power substation with electrical transformers",
    "an aerial view of a salt mine with white piles and processing equipment",

    # --- Disasters, Environmental & Edge Cases (101-120) ---
    "an aerial view of a flooded suburban neighborhood with water covering roads",
    "satellite image of a wildfire smoke plume stretching across a forest",
    "top-down view of burn scars across a hillside left by a forest fire",
    "bird's-eye view of hurricane damage to coastal buildings and beach erosion",
    "an aerial shot of an oil slick floating on the ocean surface",
    "satellite view of a landslide blocking a river in a mountainous region",
    "overhead view of a drought-stricken reservoir with exposed sediment borders",
    "an aerial view of a tornado damage path through a residential town",
    "top-down view of volcanic ash coverage across surrounding land",
    "bird's-eye view of coastal erosion carving away a cliffside edge",
    "an aerial view of a plastic waste landfill with garbage piles",
    "satellite image of algal bloom forming green swirls in a lake",
    "top-down view of a bridge collapsed into a river below",
    "an aerial view of a desert encampment or refugee tent city",
    "bird's-eye view of an emergency response staging area with red fire trucks",
    "overhead drone shot of an abandoned industrial factory overgrown with weeds",
    "an aerial perspective of an isolated lighthouse on a tiny rocky outcrop",
    "top-down satellite view of military aircraft parked on a tarmac",
    "an aerial view of a border fence separating two distinct land use zones",
    "bird's-eye view of a ship aground or wrecked along a sandy coastline"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================================================================
# 1. Load CLIP Model and Tokenizer
# =====================================================================
print(f"Loading OpenCLIP model ({CLIP_VISION_MODEL_NAME})...")
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_VISION_MODEL_NAME, 
    pretrained=CLIP_VISION_PRETRAINED, 
    device=device
)
tokenizer = open_clip.get_tokenizer(CLIP_VISION_MODEL_NAME)
model.eval()

# =====================================================================
# 2. Read Parquet and Sample N Images
# =====================================================================
print(f"Reading Parquet file: {PARQUET_FILE_PATH}")
df = pd.read_parquet(PARQUET_FILE_PATH)

total_rows = len(df)
print(f"Total images in parquet file: {total_rows}")

if SAMPLE_SIZE and SAMPLE_SIZE < total_rows:
    print(f"Sampling {SAMPLE_SIZE} images randomly (seed={RANDOM_SEED})...")
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
else:
    print(f"Using all {total_rows} images.")

# Extract raw bytes list
raw_bytes_list = df[IMAGE_COLUMN_NAME].tolist()

# =====================================================================
# 3. Process Images & Extract Embeddings in Batches
# =====================================================================
image_embeddings_list = []

print(f"Processing images and extracting features in batches of {BATCH_SIZE}...")
with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu"):
    for i in range(0, len(raw_bytes_list), BATCH_SIZE):
        batch_bytes = raw_bytes_list[i : i + BATCH_SIZE]
        batch_tensors = []

        for raw_bytes in batch_bytes:
            try:
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                batch_tensors.append(preprocess(img))
            except Exception as e:
                continue  # Skip corrupt or unreadable image bytes

        if not batch_tensors:
            continue

        # Stack batch and push to GPU
        image_batch = torch.stack(batch_tensors).to(device)
        
        # Encode & L2 Normalize
        features = model.encode_image(image_batch)
        features /= features.norm(dim=-1, keepdim=True)
        
        image_embeddings_list.append(features.cpu())

if not image_embeddings_list:
    raise ValueError("No valid images were successfully processed.")

# Combine all image embeddings: Shape [N_sampled_valid_images, Embedding_Dim]
image_features = torch.cat(image_embeddings_list, dim=0).to(device)
print(f"Successfully processed {image_features.shape[0]} valid images.")

# =====================================================================
# 4. Compute Query Embeddings & Cosine Similarity Matrix
# =====================================================================
print("Extracting text embeddings...")
with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu"):
    text_tokens = tokenizer(QUERIES).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute Cosine Similarity Matrix: Shape [N_queries, N_images]
    # Convert directly to float32 upon moving to CPU
    similarity_matrix = (text_features @ image_features.T).to(torch.float32).cpu().numpy()

# Flatten scores for all (query x image) pairs
similarity_scores = similarity_matrix.flatten().astype(np.float64)

# =====================================================================
# 5. Report Statistical Properties
# =====================================================================
def report_distribution_properties(scores):
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    median_val = np.median(scores)
    iqr_val = stats.iqr(scores)
    skewness = stats.skew(scores)
    kurt = stats.kurtosis(scores)
    
    print("\n" + "="*50)
    print("      TEXT-TO-IMAGE SIMILARITY DISTRIBUTION")
    print("="*50)
    print(f"Total Sampled Images    : {image_features.shape[0]}")
    print(f"Total Query-Image Pairs : {len(scores)}")
    print(f"Mean Similarity         : {mean_val:.4f}")
    print(f"Std Deviation           : {std_val:.4f}")
    print(f"Median Similarity       : {median_val:.4f}")
    print(f"Interquartile Range(IQR): {iqr_val:.4f}")
    print(f"Min Score               : {np.min(scores):.4f}")
    print(f"Max Score               : {np.max(scores):.4f}")
    print(f"Skewness                : {skewness:.4f} ('> 0' -> right-tailed)")
    print(f"Kurtosis                : {kurt:.4f}")
    print("="*50 + "\n")

report_distribution_properties(similarity_scores)

# =====================================================================
# 6. Plot Similarity Score Distribution
# =====================================================================
plt.figure(figsize=(10, 6))

# Histogram
count, bins, _ = plt.hist(
    similarity_scores, 
    bins=30, 
    density=True, 
    alpha=0.6, 
    color='#2b5c8f', 
    edgecolor='black', 
    label='Score Frequency'
)

# Kernel Density Estimate Curve
kde = stats.gaussian_kde(similarity_scores)
x_grid = np.linspace(min(similarity_scores) - 0.05, max(similarity_scores) + 0.05, 200)
plt.plot(x_grid, kde(x_grid), color='#d95f02', linewidth=2.5, label='KDE Curve')

# Reference Lines
plt.axvline(np.mean(similarity_scores), color='red', linestyle='--', linewidth=1.5, label=f'Mean ({np.mean(similarity_scores):.3f})')
plt.axvline(np.median(similarity_scores), color='green', linestyle=':', linewidth=1.5, label=f'Median ({np.median(similarity_scores):.3f})')

plt.title(
    f"Text-to-Image Cosine Similarity Scores (N = {image_features.shape[0]} sampled images)\n"
    f"Model: {CLIP_VISION_MODEL_NAME} ({CLIP_VISION_PRETRAINED})", 
    fontsize=12, 
    fontweight='bold'
)
plt.xlabel("Cosine Similarity Score", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()