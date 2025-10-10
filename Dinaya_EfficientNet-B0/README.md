# Dinaya's Component — EfficientNet-B0 (Chest X-ray: NORMAL vs PNEUMONIA)

This is my isolated sandbox for the EfficientNet-B0 model. After review we can promote
the shared bits into the team `/src` and `/configs`.

## Folder layout
- `notebooks/00_eda.ipynb` — quick class counts + sample grid
- `src/` — minimal training/eval code for EfficientNet-B0
- `configs/efficientnet_b0.yaml` — run settings
- `checkpoints/`, `runs/` — outputs (gitignored)
