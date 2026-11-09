"""Run pinned Scaffold training only, in its own Python environment.

Upstream optimizer/model functions are retained. The launcher skips the GPU
auto-selection preamble, LPIPS evaluation setup, viewer and final evaluation.
Compatible with the Python 3.7 environment supplied by upstream.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys


def training_module(source):
    tree = ast.parse(source)
    torch_import = next((node.lineno for node in tree.body
                         if isinstance(node, ast.Import)
                         and any(alias.name == "torch" for alias in node.names)), None)
    if torch_import is None:
        raise ValueError("Unsupported Scaffold train.py: missing top-level torch import")
    retained = []
    for node in tree.body:
        # The upstream preamble runs nvidia-smi/grep and changes GPU selection.
        if node.lineno < torch_import and not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        # LPIPS is only used by upstream's final evaluation. Loading it would
        # allocate a VGG model on CUDA and may download weights unnecessarily.
        if isinstance(node, ast.Import):
            node.names = [alias for alias in node.names if alias.name != "lpips"]
            if not node.names:
                continue
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "lpips_fn"
                for target in node.targets):
            continue
        retained.append(node)
    tree.body = retained
    return ast.fix_missing_locations(tree)


def main(argv=None):
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument("--repository", type=Path, required=True)
    selected, remaining = preparser.parse_known_args(argv)
    repository = selected.repository.resolve()
    sys.path.insert(0, str(repository))
    source = repository / "train.py"
    namespace = {"__name__": "_scaffold_training_only", "__file__": str(source)}
    exec(compile(training_module(source.read_text(encoding="utf-8")),
                 str(source), "exec"), namespace)
    required = ("ModelParams", "OptimizationParams", "PipelineParams",
                "training", "get_logger", "safe_state", "network_gui")
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError("Unsupported Scaffold training interface: " + ", ".join(missing))
    parser = argparse.ArgumentParser(description=__doc__)
    model = namespace["ModelParams"](parser)
    optimization = namespace["OptimizationParams"](parser)
    pipeline = namespace["PipelineParams"](parser)
    args = parser.parse_args(remaining)
    if args.iterations <= 0 or not args.eval:
        raise ValueError("Positive iterations and --eval are required")
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    logger = namespace["get_logger"](args.model_path)
    namespace["safe_state"](False)
    namespace["network_gui"].conn = None
    namespace["network_gui"].try_connect = lambda: None
    # Preserve training itself, save only the fixed final checkpoint and avoid
    # test-image evaluation or post-training rendering in this preparation stage.
    namespace["training"](
        model.extract(args), optimization.extract(args), pipeline.extract(args),
        Path(args.source_path).name, [], [args.iterations], [], None, -1,
        wandb=None, logger=logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
