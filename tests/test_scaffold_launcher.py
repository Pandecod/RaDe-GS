"""Exercise the real Scaffold launcher with namespace-package collisions, without CUDA."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "research_pipeline/scaffold_train.py"


class ScaffoldLauncherTests(unittest.TestCase):
    def fixture(self, root):
        repository = root / "scaffold"
        (repository / "utils").mkdir(parents=True)
        # Match upstream: utils is a namespace directory without __init__.py.
        (repository / "utils/general_utils.py").write_text("TOKEN = 'scaffold utility'\n")
        (repository / "utils/loss_utils.py").write_text(
            "from .general_utils import TOKEN\n")
        (repository / "torch.py").write_text("# CPU-only fixture.\n")
        # A dependency can trigger the first utils import before train.py does.
        (repository / "wandb.py").write_text("from utils.loss_utils import TOKEN\n")
        script = textwrap.dedent("""\
            import torch
            import wandb
            import json
            from pathlib import Path
            from types import SimpleNamespace
            from utils.loss_utils import TOKEN
            import utils.loss_utils as selected_utility
            class ModelParams:
                def __init__(self, parser):
                    parser.add_argument('-s', dest='source_path')
                    parser.add_argument('-m', dest='model_path')
                    parser.add_argument('--eval', action='store_true')
                def extract(self, args):
                    return args
            class OptimizationParams:
                def __init__(self, parser):
                    parser.add_argument('--iterations', type=int)
                def extract(self, args):
                    return args
            class PipelineParams:
                def __init__(self, parser):
                    pass
                def extract(self, args):
                    return args
            network_gui = SimpleNamespace(conn=None, try_connect=lambda: None)
            def safe_state(quiet):
                assert quiet is False
            def get_logger(path):
                return None
            def training(dataset, opt, pipe, name, tests, saves, checks, checkpoint,
                         debug, wandb=None, logger=None):
                assert TOKEN == 'scaffold utility'
                assert dataset.eval and tests == [] and saves == [opt.iterations]
                assert checks == [] and checkpoint is None and wandb is None
                receipt = {'utility': str(Path(selected_utility.__file__).resolve()),
                           'iterations': opt.iterations, 'split': 'held-out'}
                (Path(dataset.model_path) / 'receipt.json').write_text(json.dumps(receipt))
            if __name__ == '__main__':
                raise AssertionError('Do not run upstream post-training evaluation')
            """)
        (repository / "train.py").write_text(script)
        return repository

    def test_script_entry_point_resolves_upstream_namespace_from_both_working_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = self.fixture(root)
            for index, cwd in enumerate((REPO, repository)):
                with self.subTest(cwd=cwd):
                    output = root / ("output_" + str(index))
                    process = subprocess.run(
                        [sys.executable, '-B', str(LAUNCHER), '--repository', str(repository),
                         '-s', str(root / 'input'), '-m', str(output),
                         '--eval', '--iterations', '20'],
                        cwd=cwd, capture_output=True, text=True)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    receipt = json.loads((output / 'receipt.json').read_text())
                    self.assertEqual(Path(receipt['utility']), repository / 'utils/loss_utils.py')
                    self.assertEqual(receipt['iterations'], 20)

    def test_unrelated_utils_module_on_pythonpath_cannot_replace_scaffold_utilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = self.fixture(root)
            shadow = root / 'other-library'
            shadow.mkdir()
            (shadow / 'utils.py').write_text(
                "raise AssertionError('Unrelated utils module was selected')\n")
            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join([str(shadow), str(REPO)])
            output = root / 'output'
            process = subprocess.run(
                [sys.executable, '-B', str(LAUNCHER), '--repository', str(repository),
                 '-s', str(root / 'input'), '-m', str(output), '--eval', '--iterations', '20'],
                cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            receipt = json.loads((output / 'receipt.json').read_text())
            self.assertEqual(Path(receipt['utility']), repository / 'utils/loss_utils.py')

    def test_preflight_checks_actual_launcher_imports_without_training(self):
        from research_pipeline.scaffold_environment import probe_training_imports
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = self.fixture(root)
            before = {path.relative_to(root): path.read_bytes()
                      for path in root.rglob("*") if path.is_file()}
            result = probe_training_imports(sys.executable, repository)
            self.assertTrue(result["ready"], result)
            self.assertEqual(before, {path.relative_to(root): path.read_bytes()
                                     for path in root.rglob("*") if path.is_file()})
            train = repository / "train.py"
            train.write_text(train.read_text().replace(
                "import torch", "import torch\nimport missing_scaffold_dependency_for_regression", 1))
            result = probe_training_imports(sys.executable, repository)
            self.assertFalse(result["ready"])
            self.assertIn("missing_scaffold_dependency_for_regression", result["error"])

    def test_upstream_training_function_and_python37_launcher_syntax_are_preserved(self):
        from research_pipeline.scaffold_train import training_module
        upstream = REPO / 'eksperimen/Scaffold-GS/train.py'
        if not upstream.is_file():
            self.skipTest('Scaffold submodule is not initialized')
        source = upstream.read_text(encoding='utf-8')
        def training(tree):
            return next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'training')
        self.assertEqual(ast.dump(training(ast.parse(source))),
                         ast.dump(training(training_module(source))))
        ast.parse(LAUNCHER.read_text(encoding='utf-8'), feature_version=(3, 7))


if __name__ == '__main__':
    unittest.main()
