"""Check configured output storage with small files before a costly GPU run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.config import REPO, load_config, resolve_path
from research_pipeline.scaffold_initialization import prepare_dataset
from research_pipeline.utils import atomic_write, read_json, write_json


def require_mounted_drive(value: str) -> None:
    match = re.match(r"^/mnt/([a-zA-Z])(?:/|$)", value)
    if match:
        mount = Path('/mnt') / match.group(1).lower()
        if not os.path.ismount(mount):
            raise ValueError(f"Drive {match.group(1).upper()} belum terpasang pada {mount}; jalankan dari Ubuntu WSL dan periksa mount sebelum membuat output")


def check_storage(config_paths) -> list[dict]:
    targets = []
    for config_path in config_paths:
        config = load_config(Path(config_path))
        for value in (config['output']['root'], config['results']['csv']):
            require_mounted_drive(value)
        targets.extend([resolve_path(config['output']['root']),
                        resolve_path(config['results']['csv']).parent])
    targets = list(dict.fromkeys(targets))
    results = []
    # Source stays beside the Linux checkout, independently of TMPDIR settings.
    with tempfile.TemporaryDirectory(prefix='.storage_source_', dir=REPO) as source_tmp:
        source = Path(source_tmp) / 'scene'
        sparse = source / 'sparse/0'
        sparse.mkdir(parents=True)
        (source / 'images').mkdir()
        payload = b'small storage probe; not an actual training image'
        (source / 'images/probe.jpg').write_bytes(payload)
        for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
            (sparse / name).write_bytes(b'storage probe')
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)
            # Only newly created, unique probe directories are removed on exit.
            with tempfile.TemporaryDirectory(prefix='.storage_probe_', dir=target) as probe_tmp:
                probe = Path(probe_tmp)
                metadata = probe / 'probe.json'
                write_json(metadata, {'version': 1})
                write_json(metadata, {'version': 2})
                if read_json(metadata) != {'version': 2}:
                    raise OSError(f'Atomic JSON replacement failed: {target}')
                csv = probe / 'probe.csv'
                atomic_write(csv, lambda stream: stream.write('stage,status\nprobe,ok\n'))
                if csv.read_text(encoding='utf-8') != 'stage,status\nprobe,ok\n':
                    raise OSError(f'CSV readback failed: {target}')
                # Exercise the actual Scaffold hardlink/copy fallback and copy2 metadata.
                staged = probe / 'staged'
                prepare_dataset(source, staged)
                if (staged / 'images/probe.jpg').read_bytes() != payload:
                    raise OSError(f'Scaffold input copy failed: {target}')
                for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
                    if (staged / 'sparse/0' / name).read_bytes() != b'storage probe':
                        raise OSError(f'COLMAP metadata copy failed: {target}')
            free_gib = shutil.disk_usage(target).free / (1024 ** 3)
            results.append({'path': str(target), 'free_gib': free_gib})
            print(f'OK: {target} | ruang kosong {free_gib:.2f} GiB')
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', nargs='+', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        check_storage(args.config)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'Pemeriksaan penyimpanan gagal: {error}', file=sys.stderr)
        return 1
    print('Pemeriksaan tulis/replace/sync/copy lolos. Ini bukan pengukuran kecepatan atau jaminan kapasitas untuk seluruh run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
