# Research methodology notes

The methodology is:

```text
Dataset Togog Bali images
        ↓
COLMAP preprocessing
        ↓
COLMAP dataset (images/ and sparse/0/)
        ↓
RaDe-GS Research Pipeline
        ↓
RaDe-GS training
        ↓
Rendering
        ↓
Image evaluation (SSIM, PSNR, LPIPS)
        ↓
Resource guard and mesh extraction
        ↓
Optional geometry evaluation
        ↓
Per-scene records and experiment_results.csv
```

COLMAP preprocessing is explicitly outside and before this pipeline. Dataset
folders supplied to the pipeline must already contain a valid COLMAP model.

The research model is RaDe-GS only. The pipeline does not alter the RaDe-GS
algorithm. It automates execution, process monitoring, evaluation, failure
recovery, and result collection around the repository's original entry points.

Scenes run sequentially so training, rendering, evaluation, and mesh extraction
for one object finish or fail before the next object begins. The folder name is
used consistently as `scene_name` and `object_name`. The fixed mesh parameters
remain identical across objects, including after OOM, so results from different
objects remain comparable.

Raw logs are the source of truth. Parsed metrics are best-effort derivatives;
format changes or missing log values produce null fields rather than invented
measurements. Reproducibility metadata and a configuration-derived experiment
ID distinguish research configurations and support safe resume behavior.
