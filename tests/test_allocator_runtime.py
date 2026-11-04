"""Exercise allocator startup ordering with CPU mocks, without importing train.py."""
import ast
from argparse import ArgumentParser
import contextlib
import hashlib
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_pipeline.config import REPO, load_config, repository_parser


class AllocatorRuntimeTests(unittest.TestCase):
    def tree(self):
        return ast.parse((REPO / "train.py").read_text(encoding="utf-8"))

    def test_bootstrap_configures_cache_before_torch_and_preserves_opt_out(self):
        tree = self.tree()
        torch_index = next(i for i, node in enumerate(tree.body)
                           if isinstance(node, ast.Import) and any(a.name == "torch" for a in node.names))
        prefix = ast.Module(body=tree.body[:torch_index], type_ignores=[])
        for as_main, enabled in ((True, True), (True, False), (False, True)):
            with self.subTest(main=as_main, gc=enabled):
                argv = ["train.py", "--iterations", "30000", "--resolution", "1920", "--eval"]
                if enabled:
                    argv.append("--enable_allocator_gc")
                with patch.object(sys, "argv", argv), patch.dict(
                        os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "existing_setting"}):
                    exec(compile(prefix, "train.py", "exec"),
                         {"__name__": "__main__" if as_main else "train"})
                    expected = ("backend:native,garbage_collection_threshold:0.8"
                                if as_main and enabled else "existing_setting")
                    self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], expected)

    def test_explicit_fraction_is_opt_in_and_set_before_training(self):
        main = self.tree().body[-1]
        branch_index, branch = next(
            (i, n) for i, n in enumerate(main.body)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
            and isinstance(n.test.value, ast.Name) and n.test.value.id == "args"
            and n.test.attr == "enable_allocator_gc")
        training_index = next(
            i for i, n in enumerate(main.body)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name) and n.value.func.id == "training")
        self.assertLess(branch_index, training_index)
        for enabled in (False, True):
            with self.subTest(gc=enabled):
                cuda = SimpleNamespace(set_per_process_memory_fraction=Mock(),
                                       get_per_process_memory_fraction=Mock(return_value=1.0))
                args = SimpleNamespace(enable_allocator_gc=enabled)
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    exec(compile(ast.Module(body=[branch], type_ignores=[]), "train.py", "exec"),
                         {"args": args, "torch": SimpleNamespace(cuda=cuda)})
                if enabled:
                    cuda.set_per_process_memory_fraction.assert_called_once_with(1.0)
                    cuda.get_per_process_memory_fraction.assert_called_once_with()
                    self.assertEqual(args.allocator_memory_fraction, 1.0)
                    self.assertIn("native GC threshold=0.8", printed.getvalue())
                else:
                    cuda.set_per_process_memory_fraction.assert_not_called()
                    cuda.get_per_process_memory_fraction.assert_not_called()

    def test_flag_leaves_other_training_defaults_unchanged(self):
        parser = repository_parser("training")
        before = vars(parser.parse_args([]))
        after = vars(parser.parse_args(["--enable_allocator_gc"]))
        self.assertFalse(before.pop("enable_allocator_gc"))
        self.assertTrue(after.pop("enable_allocator_gc"))
        self.assertEqual(before, after)

    def test_training_algorithm_still_matches_original_baseline(self):
        function = next(n for n in self.tree().body if isinstance(n, ast.FunctionDef)
                        and n.name == "training")
        digest = hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, 'd95fedb8c72ca6c30a1f03e2af42c7810f1842a64ea4f618204c108cff585d61')

    def test_shipped_presets_enable_allocator_without_internal_profiling(self):
        for name in ("togog_bali", "tanks_temples", "togog_bali_1600", "togog_bali_full_gc"):
            with self.subTest(config=name):
                config = load_config(REPO / "configs" / (name + ".yaml"))
                self.assertIn("--enable_allocator_gc", config["training"]["extra_args"])
                self.assertNotIn("--profile_training", config["training"]["extra_args"])
                args = repository_parser("training").parse_args(config["training"]["extra_args"])
                self.assertTrue(args.eval)
                self.assertEqual(config["mesh"]["voxel_size"], .009)
                self.assertEqual(config["mesh"]["max_depth"], 8.)
                if name in ("togog_bali", "tanks_temples"):
                    self.assertEqual(args.resolution, 1920)


if __name__ == "__main__":
    unittest.main()
