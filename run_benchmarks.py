#!/usr/bin/env python3
"""
Benchmark evaluation for steered models on VLMEvalKit datasets.

Runs all steering methods + baseline on standard VLM benchmarks and
produces a comparison table.

Benchmarks:
  - MMBench_DEV_EN_V11  (general VLM capability, MCQ)
  - MME                 (perception + cognition, Y/N)
  - MMStar              (challenging multimodal reasoning, MCQ)

Usage:
    # Full run:
    python run_benchmarks.py

    # Specific methods / benchmarks:
    python run_benchmarks.py \\
        --methods baseline geo_svd sph_svd \\
        --benchmarks MMBench_DEV_EN_V11

    # Different model / layer:
    python run_benchmarks.py \\
        --model_path Qwen/Qwen2.5-VL-7B-Instruct \\
        --layer lm-layer22

    # Evaluation only (skip inference):
    python run_benchmarks.py --mode eval

    # Dry run (generate config only):
    python run_benchmarks.py --dry_run
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Compat shim: some VLMEvalKit plugins import ``AutoModelForImageTextToText``
# which is only present in transformers>=5. In compat environments pinned to
# transformers 4.45 (e.g. for Phi-3.5-vision) we alias it to
# ``AutoModelForCausalLM`` so the ``import vlmeval.vlm`` side-effects in
# ``register_steered_class`` do not crash.
try:  # pragma: no cover - trivial import shim
    import transformers as _tf
    if not hasattr(_tf, "AutoModelForImageTextToText"):
        _tf.AutoModelForImageTextToText = _tf.AutoModelForCausalLM
except Exception:
    pass

from steering import (
    DEFAULT_MODEL_PATH,
    DEFAULT_LAYERS,
    MODEL_PRESETS,
    STEERER_FILENAMES,
)
from vlmeval_wrapper import SteeredQwen2VLChat, resolve_steerer_dir

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BENCHMARKS = [
    "MMStar",
]

WORK_DIR = "benchmark_outputs"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_vlmeval_config(
    methods: list[str],
    benchmarks: list[str],
    alphas: list[float],
    steerer_dir: str,
    model_path: str,
    layers: list[str],
    mode: str = "race",
    gate_floor: float | None = None,
) -> dict:
    """Build a VLMEvalKit-style config JSON structure.

    When *layers* has a single entry the legacy single-layer args are used.
    When multiple layers are given, ``steering_configs`` is built so that
    each method+alpha combination steers all layers simultaneously.
    """
    models: dict = {}

    if "baseline" in methods:
        models["Baseline"] = {
            "class": "SteeredQwen2VLChat",
            "model_path": model_path,
        }

    for method in methods:
        if method == "baseline":
            continue
        filename = STEERER_FILENAMES.get(method)
        if filename is None:
            print(f"WARNING: Unknown method '{method}', skipping")
            continue

        # Find which layers have this steerer
        layer_paths: dict[str, Path] = {}
        for layer in layers:
            p = resolve_steerer_dir(steerer_dir, model_path, layer, mode=mode) / filename
            if p.exists():
                layer_paths[layer] = p

        if not layer_paths:
            print(f"WARNING: Steerer '{method}' not found in any layer dir, skipping")
            continue

        for alpha in alphas:
            alpha_str = str(alpha).replace(".", "p")
            gf_tag = f"_gf{str(gate_floor).replace('.', 'p')}" if gate_floor is not None else ""
            name = f"{method}_a{alpha_str}{gf_tag}"

            if len(layer_paths) == 1:
                layer, steerer_path = next(iter(layer_paths.items()))
                cfg_entry: dict = {
                    "class": "SteeredQwen2VLChat",
                    "model_path": model_path,
                    "steerer_type": method,
                    "steerer_path": str(steerer_path.resolve()),
                    "steering_alpha": alpha,
                    "steering_layer": layer,
                }
                if gate_floor is not None:
                    cfg_entry["gate_floor"] = gate_floor
                models[name] = cfg_entry
            else:
                configs = []
                for ln, p in layer_paths.items():
                    c = {
                        "steerer_type": method,
                        "steerer_path": str(p.resolve()),
                        "layer": ln,
                        "alpha": alpha,
                    }
                    if gate_floor is not None:
                        c["gate_floor"] = gate_floor
                    configs.append(c)
                models[name] = {
                    "class": "SteeredQwen2VLChat",
                    "model_path": model_path,
                    "steering_configs": configs,
                }

    data = {bench: {} for bench in benchmarks}
    return {"model": models, "data": data}


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

def register_steered_class():
    """Register SteeredQwen2VLChat in VLMEvalKit's vlm module."""
    import vlmeval.vlm

    vlmeval.vlm.SteeredQwen2VLChat = SteeredQwen2VLChat


def _build_model(model_cfg: dict):
    import vlmeval.vlm

    cfg = dict(model_cfg)
    cls_name = cfg.pop("class")
    cls = getattr(vlmeval.vlm, cls_name)
    return cls(**cfg)


# ---------------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------------

def get_judge_kwargs(dataset_name: str, dataset) -> dict:
    judge_kwargs: dict = {"nproc": 1, "verbose": False, "retry": 3}
    if hasattr(dataset, "TYPE") and dataset.TYPE in ("MCQ", "Y/N", "MCQ_MMMU_Pro"):
        judge_kwargs["model"] = "exact_matching"
    else:
        judge_kwargs["model"] = "exact_matching"
    return judge_kwargs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _find_existing_acc(work_dir: str, model_name: str, dataset_name: str) -> str | None:
    """Search for an existing _acc.csv file across all eval_id date folders."""
    import glob
    import os.path as osp
    model_dir = osp.join(work_dir, model_name)
    if not osp.isdir(model_dir):
        return None
    pattern = osp.join(model_dir, "T*_steered", f"{model_name}_{dataset_name}_acc.csv")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _find_existing_pred(work_dir: str, model_name: str, dataset_name: str) -> str | None:
    """Search for an existing prediction file across all eval_id date folders."""
    import glob
    import os.path as osp
    model_dir = osp.join(work_dir, model_name)
    if not osp.isdir(model_dir):
        return None
    for ext in (".xlsx", ".tsv", ".csv"):
        pattern = osp.join(model_dir, "T*_steered", f"{model_name}_{dataset_name}{ext}")
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def run_benchmarks(config: dict, work_dir: str, mode: str = "all") -> dict:
    import torch
    import os.path as osp
    import pandas as pd

    register_steered_class()

    from vlmeval.smp import get_logger
    from vlmeval.inference import infer_data_job
    from vlmeval.dataset import build_dataset

    logger = get_logger("BENCHMARK")

    model_cfgs = config["model"]
    data_cfgs = config["data"]

    date = datetime.now().strftime("%Y%m%d")
    eval_id = f"T{date}_steered"

    all_results: dict = {}

    for model_idx, (model_name, model_cfg) in enumerate(model_cfgs.items()):
        logger.info(
            "\n%s\nMODEL [%d/%d]: %s\n  steerer=%s, alpha=%s\n%s",
            "=" * 70,
            model_idx + 1,
            len(model_cfgs),
            model_name,
            model_cfg.get("steerer_type", "none"),
            model_cfg.get("steering_alpha", "-"),
            "=" * 70,
        )

        # Check if all benchmarks are already done for this model
        all_done = all(
            _find_existing_acc(work_dir, model_name, ds) is not None
            for ds in data_cfgs
        )
        if all_done and mode in ("all", "infer"):
            logger.info("  SKIP %s — all benchmarks already completed", model_name)
            # Still collect eval results for the report
            for dataset_name in data_cfgs:
                acc_file = _find_existing_acc(work_dir, model_name, dataset_name)
                if acc_file:
                    try:
                        acc_df = pd.read_csv(acc_file)
                        if len(acc_df) == 1:
                            all_results.setdefault(dataset_name, {})[model_name] = acc_df.iloc[0].to_dict()
                        else:
                            all_results.setdefault(dataset_name, {})[model_name] = acc_df.to_dict()
                        logger.info("  Loaded cached results for %s x %s", model_name, dataset_name)
                    except Exception as e:
                        logger.warning("  Could not load cached acc for %s: %s", dataset_name, e)
            continue

        pred_root = osp.join(work_dir, model_name, eval_id)
        os.makedirs(pred_root, exist_ok=True)

        model = None
        need_model = False
        for dataset_name in data_cfgs:
            if _find_existing_acc(work_dir, model_name, dataset_name) is None:
                need_model = True
                break

        if need_model and mode in ("all", "infer"):
            t0 = time.time()
            try:
                model = _build_model(model_cfg)
                logger.info("  Model built in %.1fs", time.time() - t0)
            except Exception as e:
                logger.error("  FAILED to build model: %s", e)
                traceback.print_exc()
                continue

        for dataset_name in data_cfgs:
            logger.info("\n  --- %s x %s ---", model_name, dataset_name)

            existing_acc = _find_existing_acc(work_dir, model_name, dataset_name)
            if existing_acc is not None:
                logger.info("  SKIP — results already exist at %s", existing_acc)
                try:
                    acc_df = pd.read_csv(existing_acc)
                    if len(acc_df) == 1:
                        all_results.setdefault(dataset_name, {})[model_name] = acc_df.iloc[0].to_dict()
                    else:
                        all_results.setdefault(dataset_name, {})[model_name] = acc_df.to_dict()
                except Exception as e:
                    logger.warning("  Could not load cached acc: %s", e)
                continue

            dataset = build_dataset(dataset_name)
            if dataset is None:
                logger.error("  FAILED to build dataset: %s", dataset_name)
                continue

            result_file = osp.join(pred_root, f"{model_name}_{dataset_name}.xlsx")

            # Phase 1: Inference
            if mode in ("all", "infer"):
                try:
                    t1 = time.time()
                    model = infer_data_job(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=False,
                        api_nproc=1,
                    )
                    logger.info("  Inference: %.1fs", time.time() - t1)
                except Exception as e:
                    logger.error("  Inference ERROR: %s", e)
                    traceback.print_exc()
                    continue

            # Phase 2: Evaluation
            if mode in ("all", "eval"):
                if not osp.exists(result_file):
                    existing_pred = _find_existing_pred(work_dir, model_name, dataset_name)
                    if existing_pred:
                        result_file = existing_pred
                    else:
                        for ext in (".xlsx", ".tsv", ".csv"):
                            alt = osp.join(pred_root, f"{model_name}_{dataset_name}{ext}")
                            if osp.exists(alt):
                                result_file = alt
                                break
                        else:
                            logger.warning("  No prediction file found for evaluation")
                            continue

                try:
                    judge_kwargs = get_judge_kwargs(dataset_name, dataset)
                    logger.info("  Evaluating with judge=%s", judge_kwargs.get("model", "?"))
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)

                    if eval_results is not None:
                        all_results.setdefault(dataset_name, {})[model_name] = eval_results
                        if isinstance(eval_results, dict):
                            logger.info("  Results: %s", json.dumps(eval_results, indent=2))
                        elif isinstance(eval_results, pd.DataFrame):
                            logger.info("  Results:\n%s", eval_results.to_string())
                        else:
                            logger.info("  Results: %s", eval_results)
                    else:
                        logger.warning("  Evaluation returned None")
                except Exception as e:
                    logger.error("  Evaluation ERROR: %s", e)
                    traceback.print_exc()

        if model is not None:
            del model
            torch.cuda.empty_cache()
            logger.info("  Model released from GPU")

    return all_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: dict, output_path: str | None = None) -> str:
    import pandas as pd

    if not results:
        return "No results to display."

    lines = [
        "# Steered Model Benchmark Comparison\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    for bench_name, bench_results in results.items():
        lines.append(f"## {bench_name}\n")
        rows: list[dict] = []
        for model_name, result in sorted(bench_results.items()):
            if isinstance(result, dict):
                row: dict = {"Method": model_name}
                row.update(result)
                rows.append(row)
            elif isinstance(result, pd.DataFrame):
                if len(result) == 1:
                    row = {"Method": model_name}
                    row.update(result.iloc[0].to_dict())
                    rows.append(row)
                else:
                    rows.append({"Method": model_name, "Result": result.to_string()})
            else:
                rows.append({"Method": model_name, "Score": str(result)})

        if rows:
            df = pd.DataFrame(rows)
            lines.append(df.to_markdown(index=False))
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run VLMEvalKit benchmarks on steered models vs baseline"
    )
    parser.add_argument(
        "--preset",
        choices=list(MODEL_PRESETS.keys()),
        default=None,
        help=f"Model preset (overrides --model_path and --layers). Available: {list(MODEL_PRESETS.keys())}",
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Steering layer(s). Multiple layers are steered simultaneously.",
    )
    parser.add_argument("--steerer_dir", default="results")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline"] + list(STEERER_FILENAMES.keys()),
    )
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--work_dir",
        default=None,
        help=(
            "Output directory for benchmark artifacts. Default: "
            "benchmark_outputs/<model_short_name>/<layers>"
        ),
    )
    parser.add_argument(
        "--run_mode",
        choices=["all", "infer", "eval"],
        default="all",
        help="Run mode: 'all' (infer+eval), 'infer' only, or 'eval' only.",
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--save_config", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["race", "gender"],
        default="race",
        help="Debiasing mode: 'race' or 'gender'. Determines which steerers to load.",
    )
    parser.add_argument(
        "--gate_floor",
        type=float,
        default=None,
        help="Gate floor for gated steerers. Overrides stored value at inference time.",
    )
    args = parser.parse_args()

    if args.preset:
        cfg = MODEL_PRESETS[args.preset]
        args.model_path = cfg["model_path"]
        args.layers = cfg["layers"]

    if args.work_dir is None:
        model_short = Path(args.model_path).name
        layers_tag = "+".join(args.layers)
        work_dir_name = WORK_DIR if args.mode == "race" else f"{WORK_DIR}_{args.mode}"
        args.work_dir = f"{work_dir_name}/{model_short}/{layers_tag}"

    config = build_vlmeval_config(
        methods=args.methods,
        benchmarks=args.benchmarks,
        alphas=args.alphas,
        steerer_dir=args.steerer_dir,
        model_path=args.model_path,
        layers=args.layers,
        mode=args.mode,
        gate_floor=args.gate_floor,
    )

    print("=" * 70)
    print(f"STEERED MODEL BENCHMARK CONFIGURATION  (mode={args.mode})")
    print("=" * 70)
    print(f"\nModel: {args.model_path}")
    print(f"Layers: {args.layers}")
    print(f"Mode: {args.mode}")
    print(f"\nModels ({len(config['model'])}):")
    for name, cfg in config["model"].items():
        steerer = cfg.get("steerer_type", "none")
        alpha = cfg.get("steering_alpha", "-")
        print(f"  {name:30s}  steerer={steerer:20s}  alpha={alpha}")
    print(f"\nBenchmarks ({len(config['data'])}):")
    for name in config["data"]:
        print(f"  {name}")
    print(f"\nOutput dir: {args.work_dir}")
    print("=" * 70)

    config_path = args.save_config or os.path.join(args.work_dir, "benchmark_config.json")
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running benchmarks.")
        return

    print(f"\n{'=' * 70}")
    print("STARTING BENCHMARK EVALUATION")
    print(f"{'=' * 70}\n")

    t_start = time.time()
    results = run_benchmarks(config, args.work_dir, args.run_mode)
    total_time = time.time() - t_start

    report_path = args.report or os.path.join(args.work_dir, "BENCHMARK_COMPARISON.md")
    report = generate_report(results, output_path=report_path)
    print("\n" + report)
    print(f"\nTotal time: {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
