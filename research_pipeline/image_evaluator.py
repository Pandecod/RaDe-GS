"""Reuse RaDe-GS metric functions, one image pair at a time in a child process."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from research_pipeline.utils import write_json

def rendered_sets(output: Path) -> dict[str, tuple[Path, list[Path]]]:
    result = {}
    for split in ("train", "test"):
        candidates = []
        for folder in (output / split).glob("ours_*"):
            try:
                iteration = int(folder.name.removeprefix("ours_"))
            except ValueError:
                continue
            if (folder / "renders").is_dir():
                candidates.append((iteration, folder))
        if candidates:
            folder = max(candidates)[1]
            images = sorted((folder / "renders").glob("*.png"))
            result[split] = (folder, images)
    return result

def render_statistics(output: Path, duration: float) -> dict:
    count = sum(len(images) for _, images in rendered_sets(output).values())
    return {"image_count": count, "seconds_per_image": duration / count if count else None,
            "fps": count / duration if count and duration > 0 else None}

def evaluation_pairs(output: Path) -> tuple[str, Path, list[tuple[Path, Path]]]:
    sets = rendered_sets(output)
    split = next((s for s in ("test", "train") if s in sets and sets[s][1]), None)
    if split is None:
        raise ValueError("No rendered PNG images available for evaluation")
    folder, images = sets[split]
    pairs = []
    for rendered in images:
        gt = folder / "gt" / rendered.name
        if not gt.is_file():
            raise ValueError(f"Missing ground truth image: {gt}")
        pairs.append((rendered, gt))
    return split, folder, pairs

def evaluate(output: Path, scene_name: str) -> dict:
    # Imports occur only inside this dedicated worker, never in planning/controller.
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch.modules.lpips import LPIPS
    split, folder, pairs = evaluation_pairs(output)
    criterion = LPIPS(net_type="vgg", version="0.1").cuda().eval()
    totals = dict.fromkeys(("ssim", "psnr", "lpips"), 0.0)
    warnings = []
    with torch.no_grad():
        for rendered, gt in pairs:
            with Image.open(rendered) as image:
                a = to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()
            with Image.open(gt) as image:
                b = to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()
            if a.shape != b.shape or a.shape[1] != 3:
                raise ValueError(f"Image shapes are incompatible: {rendered.name}")
            values = {"ssim": ssim(a, b).item(), "psnr": psnr(a, b).mean().item(),
                      "lpips": criterion(a, b).item()}
            for name, value in values.items():
                if math.isnan(value) or (math.isinf(value) and (name != "psnr" or value < 0)):
                    raise ValueError(f"Invalid {name} for {rendered.name}: {value}")
                totals[name] += value
            del a, b
    means = {name: value / len(pairs) for name, value in totals.items()}
    if math.isinf(means["psnr"]):
        means["psnr"] = None
        warnings.append("PSNR is positive infinity (perfect image match); stored as null for strict JSON")
    result = {"scene_name": scene_name, "object_name": scene_name, **means,
              "image_count": len(pairs), "split": split, "method": folder.name,
              "warnings": warnings}
    write_json(output / "research/image_metrics.json", result)
    print(f"Evaluated {len(pairs)} {split} images: {means}")
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-name", required=True)
    args = parser.parse_args()
    evaluate(args.output, args.scene_name)

if __name__ == "__main__":
    main()
