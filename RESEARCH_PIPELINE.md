# RaDe-GS Research Pipeline

This repository adds a serial orchestration layer around the existing RaDe-GS
programs. It does not replace or alter the RaDe-GS training, renderer, Gaussian
representation, CUDA rasterizer, or mesh algorithm.

## Installation

Install RaDe-GS as described in `README.md`, then install only the extra pipeline
dependencies:

```bash
pip install -r requirements-research.txt
```

`trimesh` is already present in the RaDe-GS requirements. The pipeline adds
PyYAML, psutil, and SciPy.

## Dataset layout

COLMAP preprocessing happens before this pipeline. Every direct child directory
under `dataset.root` is one scene and its folder name is the scene/object name:

```text
/home/pande/dataset_togog_bali/
  togog_arjuna/
    images/
    sparse/0/cameras.bin
    sparse/0/images.bin
    sparse/0/points3D.bin
  togog_ganesha/
    images/
    sparse/0/cameras.txt
    sparse/0/images.txt
    sparse/0/points3D.txt
```

Binary COLMAP data is accepted. A complete text camera/image pair plus
`points3D.txt` is also accepted. Invalid scene directories are reported and do
not stop later scenes.

## Configuration and commands

Edit `configs/togog_bali.yaml`, especially `dataset.root` and `output.root`.
Paths may be absolute or relative to the repository root.

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --list-scenes
python research_pipeline/run_all.py --config configs/togog_bali.yaml --dry-run
python research_pipeline/run_all.py --config configs/togog_bali.yaml
python research_pipeline/run_all.py --config configs/togog_bali.yaml --scene togog_ganesha
python research_pipeline/run_all.py --config configs/togog_bali.yaml --resume
python research_pipeline/run_all.py --config configs/togog_bali.yaml --force
```

`--resume` skips only a row whose deterministic experiment ID matches and whose
status is `success`. Failed experiments run again. `--force` always runs and
never deletes an old model, checkpoint, mesh, or log. If a scene output already
contains files, a new `runs/<timestamp>_<id>/` attempt is used.

The pipeline processes scenes in alphabetic order using one synchronous
subprocess at a time. Training uses exactly:

```bash
python train.py -s <scene> -m <output>
```

Only `training.extra_args` can add training arguments. Rendering invokes the
repository's `render.py`. Image evaluation reuses the repository SSIM and PSNR
functions and its VGG LPIPS implementation. When a test render has images it is
evaluated; otherwise the training render is evaluated, matching this checkout's
`metric.py` behavior for the default non-`--eval` training configuration.

Mesh extraction always uses the verified experiment command:

```bash
python mesh_extract.py -m <output> -r 2 --num_cluster 8 --depth_ratio 0.5
```

Configuration validation rejects attempts to override these parameters. An OOM
is recorded as `mesh_extraction_oom`; the pipeline does not retry with different
mesh parameters.

## Resource monitoring and OOM protection

The process monitor samples total RSS for the subprocess tree with psutil and
process-attributed VRAM with `nvidia-smi`. Monitoring failures do not terminate
training/rendering/mesh; unavailable values remain empty and a warning is saved.

Before mesh extraction, the resource guard waits for every prior subprocess to
exit, applies the configured cooldown, and checks available RAM and VRAM. If
safe memory cannot be verified or thresholds remain unmet through
`max_wait_seconds`, mesh extraction is not started and the scene records
`mesh_resource_check`.

## Results

RaDe-GS keeps its original output layout. Research data is added below each
attempt:

```text
<scene_output>/
  cfg_args
  point_cloud/
  train/
  test/
  recon_raw.ply
  recon_post.ply
  research/
    logs/training.log
    logs/rendering.log
    logs/image_metrics.log
    logs/mesh_extraction.log
    logs/geometry_evaluation.log
    training_metrics.csv
    image_metrics.json
    mesh_metrics.json
    geometry_metrics.json
    summary.json
```

Raw subprocess output is preserved. Structured files contain parsed training,
image, mesh, timing, RAM, VRAM, and optional geometry metrics. Missing values are
stored as JSON null or empty CSV cells, never fabricated as zero.

The master CSV path is configured by `results.csv`. It is written through a
temporary file, flushed, fsynced where supported, and replaced atomically. A
master CSV write error is critical and stops the batch. Other scene failures are
recorded before the next scene starts.

The experiment ID is SHA-256 over canonical JSON containing the scene name and
training, rendering, mesh, and geometry configuration. Each row also records the
Git commit/dirty state, Python/PyTorch/CUDA versions, GPU name, and driver when
available.

## Optional geometry evaluation

Enable `geometry_evaluation` and set `gt_root`. The pipeline looks for
`<gt_root>/<scene_name>.ply`. It samples deterministically and computes Euclidean
Chamfer distance as the mean reconstruction-to-GT distance plus the mean
GT-to-reconstruction distance using `scipy.spatial.cKDTree`. Precision and
recall use strict `distance < distance_threshold`; F1 is their harmonic mean.
When the threshold is null, Chamfer is still calculated and the other metrics
remain null with a warning.

## Dry run and failures

Dry run validates the configuration and every dataset, builds all commands, and
prints them. It imports no model/CUDA code, starts no subprocess, and writes no
result or fake metric. Stage failures are recorded in `summary.json` and the
master CSV. Supported stages include dataset validation, training, rendering,
image metrics, mesh resource check, mesh extraction/OOM, geometry evaluation,
and result recording.
