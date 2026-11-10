"""CPU checks for queue ordering, planning, and resume forwarding."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from research_pipeline.config import DEFAULTS, REPO, load_config
from research_pipeline.results import experiment_id
from research_pipeline.run_all import main
from test_research_pipeline import make_scene


NAMES = [
    'detya', 'dewa_brahma', 'dewa_sambhu', 'dewi_gangga', 'ganesha',
    'merdah', 'ngawasari', 'om_swastyastu_luh', 'om_swastyastu_muani',
    'paksi_singa', 'penabuh_cengceng', 'penabuh_gangsa', 'rangda', 'tualen',
]
TAIL = ['dewi_gangga', 'detya']
EXPECTED = [name for name in NAMES if name not in TAIL] + TAIL


class SceneOrderTests(unittest.TestCase):
    def fixture(self, root):
        config = load_config(REPO / 'configs/togog_bali_scaffold_init.yaml')
        config['dataset']['root'] = str(root / 'dataset')
        config['output']['root'] = str(root / 'output')
        config['results']['csv'] = str(root / 'results.csv')
        for name in NAMES:
            make_scene(root / 'dataset', name)
        path = root / 'config.yaml'
        path.write_text(yaml.safe_dump(config), encoding='utf-8')
        return path

    def test_default_order_stays_alphabetical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['--config', str(path), '--list-scenes']), 0)
            lines = output.getvalue().splitlines()[1:]
            self.assertEqual([line.split('. ', 1)[1] for line in lines], NAMES)

    def test_list_and_dry_run_share_the_tail_order_without_execution_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            for mode in ('--list-scenes', '--dry-run'):
                with self.subTest(mode=mode), \
                     patch('subprocess.Popen', side_effect=AssertionError('No process')), \
                     patch('research_pipeline.runner.run_batch', side_effect=AssertionError('No execution')), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(['--config', str(path), '--last-scenes', *TAIL, mode]), 0)
                text = output.getvalue()
                if mode == '--list-scenes':
                    actual = [line.split('. ', 1)[1] for line in text.splitlines()[1:]]
                    self.assertEqual(actual, EXPECTED)
                else:
                    actual = [line.split('] ', 1)[1] for line in text.splitlines() if line.startswith('[')]
                    self.assertEqual(actual, EXPECTED)
                    self.assertEqual(text.count('SCAFFOLD TRAIN:'), 14)
                    self.assertEqual(text.count('IMAGE METRICS:'), 14)
                    self.assertEqual(text.count('MESH:'), 14)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes()
                                      for p in root.rglob('*') if p.is_file()})
            self.assertFalse((root / 'output').exists())

    def test_order_reaches_real_batch_entry_point_with_resume_and_unchanged_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            original = load_config(path)
            identities = {name: experiment_id(name, original) for name in NAMES}
            seen = []
            def batch(config, scenes, **options):
                seen.append(([scene.name for scene in scenes], options))
                self.assertEqual(config, original)
                self.assertEqual(identities, {scene.name: experiment_id(scene.name, config)
                                              for scene in scenes})
                return 0
            with patch('research_pipeline.runner.run_batch', side_effect=batch):
                self.assertEqual(main(['--config', str(path), '--last-scenes', *TAIL,
                                       '--start-from', 'training', '--resume']), 0)
            self.assertEqual(seen, [(EXPECTED, {'resume': True, 'force': False, 'start_from': 'training'})])

    def test_unknown_or_duplicate_tail_stops_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            for names in (['missing_scene'], ['detya', 'detya']):
                with self.subTest(names=names), patch('research_pipeline.runner.run_batch') as batch:
                    self.assertEqual(main(['--config', str(path), '--last-scenes', *names]), 2)
                    batch.assert_not_called()
            self.assertFalse((root / 'output').exists())

    def test_single_scene_cannot_be_combined_with_tail_order(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--config', 'unused.yaml', '--scene', 'ganesha', '--last-scenes', *TAIL])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
