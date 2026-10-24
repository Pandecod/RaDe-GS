"""Direct child directories are scenes, including invalid ones."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Scene:
    name: str
    path: Path

def discover(root: Path, selected: str | None = None) -> list[Scene]:
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    scenes = [Scene(p.name, p.resolve()) for p in sorted(root.iterdir(), key=lambda p: p.name)
              if p.is_dir()]
    if selected is not None:
        scenes = [s for s in scenes if s.name == selected]
        if not scenes:
            raise ValueError(f"Scene not found among direct child directories: {selected}")
    if not scenes:
        raise ValueError(f"No scene directories found in {root}")
    return scenes

def validation_errors(scene: Scene) -> list[str]:
    errors = []
    if not (scene.path / "images").is_dir():
        errors.append("Missing images/ directory")
    sparse = scene.path / "sparse/0"
    if not sparse.is_dir():
        errors.append("Missing sparse/0/ directory")
    camera_pair = any(all((sparse / (name + suffix)).is_file() for name in ("cameras", "images"))
                      for suffix in (".bin", ".txt"))
    if not camera_pair:
        errors.append("Missing matching cameras/images .bin or .txt pair in sparse/0")
    if not any((sparse / ("points3D" + suffix)).is_file() for suffix in (".bin", ".txt")):
        errors.append("Missing points3D.bin or points3D.txt in sparse/0")
    return errors


def start_validation_errors(scene: Scene, start_from: str) -> list[str]:
    if start_from == "training":
        return validation_errors(scene)
    if start_from == "colmap":
        from research_pipeline.colmap_preprocessor import raw_input_errors
        return raw_input_errors(scene)
    raise ValueError(f"Unsupported start stage: {start_from}")
