## ST-2 Injection (ST2I)

ST-2 Injection (**ST2I**) is a video style transfer system built on top of Stable Diffusion and the StyleID method, extended with temporal consistency via optical flow (GMFlow). It supports:

- **Single-video style transfer** with temporal consistency (`run_styleid_v2v.py` – referred to here as the **ST2I video script**).
- **Batch video style transfer** over all content/style combinations (`run_styleid_v2v_batch.py` or `run_all.sh`).
- **Baseline StyleID image/video stylization** without temporal fusion (code under `styleid/`).
- **Evaluation utilities** for aesthetics, FVD, style similarity, and temporal consistency (`evaluation/`).

### Project Structure

- **`run_styleid_v2v.py`**: Main **ST2I video script** – stylizes a single content video with a single style image using `StyleIDVideoPipeline`.
- **`run_styleid_v2v_batch.py`**: Batch ST2I over all content folders and style images.
- **`run_all.sh`**: Bash wrapper that loops over content/style pairs and calls the ST2I video script.
- **`styleid_v2v/`**: Implementation of `StyleIDVideoPipeline` (video extension of `StyleIDPipeline` with GMFlow-based temporal fusion).
- **`styleid/`**: Original StyleID image pipeline (`StyleIDPipeline`) and a baseline video driver (`run_video_styleid.py`).
- **`evaluation/`**: Scripts and helpers for quantitative evaluation (FVD, aesthetics, style similarity, temporal consistency).
- **`GMflow/`**: GMFlow optical flow model code (expects weights at `gmflow/pretrained/gmflow_sintel-0c07dcb3.pth`).
- **`configs/`**: Configuration files (currently minimal/empty).

Where this README says **“ST2I video script”**, it refers to the `run_styleid_v2v.py` entry point (kept under that filename for compatibility).

### Environment & Installation

- **Python**: 3.10+ (recommended)
- **CUDA GPU**: Required for practical performance

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure you have the Stable Diffusion weights available via Hugging Face (e.g. `runwayml/stable-diffusion-v1-5`) and that GMFlow weights are placed at:

```bash
gmflow/pretrained/gmflow_sintel-0c07dcb3.pth
```

### Data Layout

ST2I expects a simple directory layout under a chosen `--data_root`:

```bash
data/
  content/
    <video_name>/
      0001.png
      0002.png
      ...
  style/
    style_1.png
    style_2.jpg
    ...
```

- **Content video**: A folder of numbered frames (`.png` or `.jpg`), e.g. `data/content/car/0001.png`.
- **Style image**: A single image file in `data/style/`, e.g. `data/style/wave.png`.

### Quickstart: Single-Video ST2I

Use the **ST2I video script** (`run_styleid_v2v.py`) to style one video with one style:

```bash
python run_styleid_v2v.py \
  --data_root ./data \
  --content_name car \
  --style_name wave.png \
  --output_dir ./results \
  --model_path runwayml/stable-diffusion-v1-5 \
  --ddim_steps 50 \
  --gamma 0.75 \
  --temperature 1.5 \
  --mask_strength 1.0 \
  --fusion-strategy anchor_only \
  --fusion-start-percent 0.5 \
  --fusion-end-percent 1.0
```

This will write stylized frames to:

```bash
results/wave_stylized_car/0001.png
results/wave_stylized_car/0002.png
...
```

and save run metadata to `results/wave_stylized_car/parameters.json`.

### Quickstart: Batch ST2I Over All Content/Styles

Two options:

- **Python batch driver** (recommended for cross-platform use):

```bash
python run_styleid_v2v_batch.py \
  --data_root ./data \
  --output_dir ./results \
  --model_path runwayml/stable-diffusion-v1-5 \
  --ddim_steps 50 \
  --gamma 0.75 \
  --temperature 1.5 \
  --mask_strength 1.0 \
  --fusion-strategy anchor_only \
  --fusion-start-percent 0.5 \
  --fusion-end-percent 1.0
```

- **Bash wrapper** (Linux/macOS):

```bash
bash run_all.sh
```

Both iterate over all `<content_name>` in `data/content/` and all style images in `data/style/`, skipping combinations that already have an output folder.

### Important Parameters (ST2I Video & Batch)

- **`--ddim_steps`**: Number of DDIM inversion/sampling steps (higher = better quality, slower).
- **`--gamma`**: Query preservation strength (higher = more content structure preserved).
- **`--temperature`**: Attention temperature (higher = stronger stylization).
- **`--mask_strength`**: How strongly fusion masks influence the final blend between warped prior frames and the freshly stylized frame.
- **`--fusion-strategy`**: `"anchor_only"` or `"anchor_and_prev"`; whether to fuse only with the first (anchor) frame or also with the previous stylized frame.
- **`--fusion-start-percent` / `--fusion-end-percent`**: Portion of the denoising schedule \([0, 1]\) where fusion is active.
- **`--without_init_adain`**: If set, disables AdaIN in the initial latent (more literal content, less style coupling).

### Baseline StyleID (No Temporal Fusion)

If you want plain StyleID without temporal fusion (e.g., per-frame or single images), use the code under `styleid/`:

- **Single images**: Run `styleid/styleid_pipeline.py` as a script (see `__main__` at the bottom of that file).
- **Per-frame video baseline**:

```bash
python styleid/run_video_styleid.py \
  --content_path ./data/content \
  --style_path ./data/style \
  --output_dir ./results_baseline_styleid
```

This processes frames independently using `StyleIDPipeline`.

### Evaluation Utilities

The `evaluation/` folder contains helpers and scripts used for experiments:

- **`frechet_video_distance/`**: FVD computation scripts.
- **`improved-aesthetic-predictor/`**: Aesthetic score predictor.
- **`style_similarity/`**: Style similarity analysis.
- **`temporal_consistency/`**: Temporal consistency metrics and ablations.

These scripts are mostly standalone; see their in-folder documentation and comments for usage details.

### Notes & Naming

- The project is referred to as **ST-2 Injection (ST2I)** in this README.
- The original entry script name `run_styleid_v2v.py` is kept on disk for backward compatibility, but conceptually it is the **ST2I video script** for single-video temporal style transfer.


