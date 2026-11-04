"""Opt-in VoxelBlockGrid layout retry around the existing native mesh script."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from research_pipeline.utils import write_json

REPO = Path(__file__).resolve().parents[1]
SIGNED_INDEX_BYTES = 1 << 31


def mesh_structure_bytes(blocks: int, block_resolution: int) -> int:
    # Open3D 0.19: four Int32 entries for each voxel in each active block.
    return int(blocks) * block_resolution ** 3 * 4 * 4


class CheckedVolume:
    """Delegate integration and extraction to Open3D; inspect its buffer first."""

    def __init__(self, volume, block_resolution, report_path, metadata):
        self._volume = volume
        self._resolution = block_resolution
        self._report_path = report_path
        self._metadata = metadata

    def __getattr__(self, name):
        return getattr(self._volume, name)

    def extract_triangle_mesh(self, *args, **kwargs):
        blocks = self._volume.hashmap().size()
        capacity = self._volume.hashmap().capacity()
        buffer_bytes = mesh_structure_bytes(blocks, self._resolution)
        self._metadata.update(
            stage="before_extraction", active_voxel_blocks=blocks,
            capacity=capacity, mesh_structure_bytes=buffer_bytes)
        write_json(self._report_path, self._metadata)
        print(f"[NATIVE MESH RETRY] blocks={blocks}; capacity={capacity}; "
              f"block_resolution={self._resolution}; "
              f"mesh_structure_GiB={buffer_bytes / 2**30:.6f}", flush=True)
        if buffer_bytes > SIGNED_INDEX_BYTES:
            self._metadata["stage"] = "buffer_guard_rejected"
            write_json(self._report_path, self._metadata)
            raise RuntimeError(
                "Native mesh buffer exceeds the suspected signed 32-bit byte-offset "
                "boundary. Extraction was stopped before calling Open3D. "
                "Do not treat this attempt as a completed mesh.")
        mesh = self._volume.extract_triangle_mesh(*args, **kwargs)
        self._metadata["stage"] = "extraction_returned"
        write_json(self._report_path, self._metadata)
        return mesh


def run_native(native_args, *, block_resolution, runtime_report,
               geometry=None, open3d_version=None, run_script=None):
    if block_resolution not in (8, 16):
        raise ValueError("block_resolution must be 8 or 16")
    if geometry is None:
        import open3d as o3d
        geometry = o3d.t.geometry
        open3d_version = o3d.__version__
    if open3d_version != "0.19.0":
        raise ValueError("This buffer guard is scoped to Open3D 0.19.0")

    native = REPO / "mesh_extract.py"
    runtime_report = Path(runtime_report)
    original_class = geometry.VoxelBlockGrid
    original_argv = sys.argv
    metadata = {
        "kind": "native_mesh_layout_retry", "open3d_version": open3d_version,
        "native_script": str(native), "block_resolution": block_resolution,
        "stage": "starting",
        "note": "Allocation layout experiment; voxel spacing and native integration/"
                "extraction routines are retained. Surface equivalence is not assumed.",
    }

    def make_volume(*args, **kwargs):
        # The restored native script constructs the grid entirely by keyword.
        if args or kwargs.get("block_resolution") != 16:
            raise ValueError("Unexpected native VoxelBlockGrid constructor; refusing override")
        options = {**kwargs, "block_resolution": block_resolution}
        metadata.update(stage="constructing", voxel_size=options.get("voxel_size"),
                        initial_block_count=options.get("block_count"))
        write_json(runtime_report, metadata)
        print(f"[NATIVE MESH RETRY] Open3D={open3d_version}; "
              f"block_resolution=16 -> {block_resolution}; "
              f"voxel_size={options.get('voxel_size')}", flush=True)
        volume = original_class(**options)
        return CheckedVolume(volume, block_resolution, runtime_report, metadata)

    try:
        geometry.VoxelBlockGrid = make_volume
        sys.argv = [str(native), *native_args]
        (run_script or runpy.run_path)(str(native), run_name="__main__")
    finally:
        geometry.VoxelBlockGrid = original_class
        sys.argv = original_argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-resolution", required=True, type=int, choices=(8, 16))
    parser.add_argument("--runtime-report", required=True, type=Path)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    native_args = args.native_args
    if native_args[:1] == ["--"]:
        native_args = native_args[1:]
    if not native_args:
        parser.error("Native mesh arguments are required after --")
    run_native(native_args, block_resolution=args.block_resolution,
               runtime_report=args.runtime_report)


if __name__ == "__main__":
    main()
