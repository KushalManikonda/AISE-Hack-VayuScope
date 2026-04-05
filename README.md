# AISE-Hack-VayuScope
## Model Architecture — PM2.5 Spatiotemporal Forecasting

## Overview

This model predicts **16 future PM2.5 concentration maps** over a 140×124 spatial grid by learning from historical pollution data and meteorological variables.

Unlike simple regression, the model must capture:
- Local pollution hotspots and boundaries
- Regional transport and atmospheric spread
- Weather-driven temporal changes
- Multi-step future evolution in a single forward pass

The architecture combines **U-Net** (local spatial learning), **FNO** (global dependency modeling), and **CBAM Attention** (feature refinement) into a unified forecasting pipeline.

---

## Input Representation

```
Shape: (160, 140, 124)

160 Channels
├── 10 PM2.5 historical frames
└── 150 meteorological channels
    └── 15 variables × 10 timesteps
```

The model sees a full history of atmospheric conditions rather than a single snapshot, enabling it to learn how pollution evolves with changing weather.

---

## Architecture Flow

```
Input (160 × 140 × 124)
        │
        ▼
[1]  Initial Conv Block          # project raw channels → learned feature space
        │
        ▼
[2]  Encoder Block 1             # local patterns, hotspots
        ├── save skip 1
        ▼
[3]  Downsample
        │
        ▼
[4]  Encoder Block 2             # mid-level structures, weather-pollution interaction
        ├── save skip 2
        ▼
[5]  Downsample
        │
        ▼
[6]  Encoder Block 3             # high-level abstractions, regional behavior
        ├── save skip 3
        ▼
[7]  Bottleneck  # compressed semantic representation
        │
        ▼
[8]  FNO Block                   # global spatial dependencies via spectral filtering
        ▼
[9]  CBAM Attention              # channel + spatial feature refinement
        ▼
[10] Decoder Block 1 + skip 3   # begin spatial reconstruction
        ▼
[11] Decoder Block 2 + skip 2   # recover regional detail
        ▼
[12] Decoder Block 3 + skip 1   # restore fine local structure
        ▼
[13] Output Head (1×1 Conv)
        ▼
Output (16 × 140 × 124)
```

---

## Component Details

### Initial Conv Block
Converts raw heterogeneous inputs (PM2.5, wind, temperature, pressure, humidity) into a unified learned feature space. Without this, the varied input types cannot be jointly processed.

### U-Net Encoder (Blocks 1–3)
Three stages of convolution + downsampling progressively expand the receptive field:
- **Block 1** — captures local edges, dense hotspot regions
- **Block 2** — understands regional concentration zones
- **Block 3** — represents the overall atmospheric state

Each stage saves a **skip connection** to preserve spatial detail for the decoder.

### Bottleneck
Compressed latent space with reduced resolution but high semantic content. Ideal location for global modeling since spatial abstraction is already complete.

### FNO — Fourier Neural Operator
Standard CNNs are limited to local context. FNO overcomes this by operating in the frequency domain:

```
Spatial Features → FFT → Spectral Filtering → Inverse FFT → Globally Enriched Features
```

- **FFT** decomposes features into frequency components capturing broad wave-like patterns
- **Spectral Filtering** learns long-range interactions and atmospheric transport effects
- **Inverse FFT** returns enriched features to spatial domain

Result: the model understands both local detail and global spread simultaneously.

### CBAM Attention
Refines bottleneck features before decoding:
- **Channel Attention** — scores each feature channel, suppresses weak signals
- **Spatial Attention** — highlights important grid regions (hotspots, corridors)

Together they make the model selectively focus on the most informative signals.

### U-Net Decoder (Blocks 10–12)
Mirrors the encoder. Each stage upsamples and merges with its corresponding skip connection:
- **Block 10** — coarse reconstruction from bottleneck
- **Block 11** — recovers regional structure
- **Block 12** — restores fine boundaries and sharpness

### Output Head
A 1×1 convolution maps decoder features to exactly **16 output channels**, one per future PM2.5 map.

---

## Why Direct Multi-Step Forecasting

The model predicts all 16 future frames in a single forward pass rather than autoregressively.

| Approach | Issue |
|---|---|
| Recursive (1 step at a time) | Errors accumulate, predictions drift |
| Direct (all 16 at once) | Stable, consistent, faster inference |

---

## Architecture Summary

| Stage | Component | Role |
|---|---|---|
| 1 | Initial Conv | Feature embedding |
| 2–6 | Encoder ×3 | Multi-scale local learning |
| 7 | Bottleneck | Compressed representation |
| 8 | FNO | Global dependency modeling |
| 9 | CBAM | Attention-based refinement |
| 10–12 | Decoder ×3 | Spatial reconstruction |
| 13 | Output Head | 16-step PM2.5 forecast |

---

## Model Link

[Kaggle — VayuScope](https://www.kaggle.com/models/nuthankatta/anrf-aise-vayuscope)
