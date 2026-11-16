"""Verify destination probes preserve existing files and fail before GPU work."""
import copy
import errno
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib

import yaml
from research_pipeline.config import DEFAULTS, REPO, load_config, resolve_path

spec = importlib.util.spec_from_file_location('storage_probe_script', REPO / 'scripts/check_output_storage.py')
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class OutputStorageTests(unittest.TestCase):
    def fixture(self, root):
        config = copy.deepcopy(DEFAULTS)
        config['dataset']['root'] = str(root / 'dataset')
        config['output']['root'] = str(root / 'saved/output')
        config['results']['csv'] = str(root / 'saved/reports/results.csv')
        path = root / 'config.yaml'
        path.write_text(yaml.safe_dump(config), encoding='utf-8')
        return path

    def test_real_probe_and_cross_filesystem_copy_fallback_preserve_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            sentinel = root / 'saved/output/existing.ply'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'existing result must remain unchanged')
            report = root / 'saved/reports/results.csv'
            report.parent.mkdir(parents=True)
            report.write_bytes(b'old csv')
            with patch('subprocess.Popen', side_effect=AssertionError('No GPU subprocess')), \
                 patch('research_pipeline.scaffold_initialization.os.link',
                       side_effect=OSError(errno.EXDEV, 'cross device')), \
                 contextlib.redirect_stdout(io.StringIO()):
                results = storage.check_storage([config, config])
            self.assertEqual(len(results), 2)
            self.assertEqual(sentinel.read_bytes(), b'existing result must remain unchanged')
            self.assertEqual(report.read_bytes(), b'old csv')
            self.assertEqual(sorted(p.relative_to(root / 'saved').as_posix()
                                    for p in (root / 'saved').rglob('*') if p.is_file()),
                             ['output/existing.ply', 'reports/results.csv'])

    def test_failed_atomic_write_is_reported_and_does_not_replace_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            sentinel = root / 'saved/output/existing.ply'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'keep')
            with patch.object(storage, 'write_json', side_effect=OSError(errno.EPERM, 'denied')), \
                 contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(storage.main(['--config', str(config)]), 1)
            self.assertIn('denied', output.getvalue())
            self.assertEqual(sentinel.read_bytes(), b'keep')
            self.assertEqual(list(sentinel.parent.iterdir()), [sentinel])

    def test_missing_windows_mount_is_rejected_before_output_creation(self):
        with patch.object(storage.os.path, 'ismount', return_value=False):
            with self.assertRaisesRegex(ValueError, 'belum terpasang'):
                storage.require_mounted_drive('/mnt/x/storage_probe/output')
        with patch.object(storage.os.path, 'ismount', return_value=True):
            storage.require_mounted_drive('/mnt/x/storage_probe/output')

    def test_shipped_presets_store_models_and_reports_inside_repository(self):
        for path in sorted((REPO / 'configs').glob('*.yaml')):
            with self.subTest(config=path.name):
                config = load_config(path)
                for section, key, folder in (
                    ('output', 'root', 'output'),
                    ('results', 'csv', 'research_results'),
                ):
                    value = config[section][key]
                    self.assertFalse(Path(value).is_absolute())
                    self.assertTrue(
                        resolve_path(value).is_relative_to(REPO / folder),
                        f'{path.name}: {section}.{key} must stay inside {folder}/',
                    )


if __name__ == '__main__':
    unittest.main()
