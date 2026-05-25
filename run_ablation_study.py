#!/usr/bin/env python3
"""
Ablation study for GGSS on a single (model, task, alpha) cell.

Runs component ablation (Table a) and sensitivity sweep (Table b):
  - Component ablation: geo_svd, sph_svd, GGSS w/o gate, GGSS w/o slerp,
    GGSS w/o norm restore, GGSS full
  - Sensitivity: (kappa, g_floor) grid

Uses existing MCQ evaluation infrastructure.  Loads the model once and
swaps steerers between runs for efficiency.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

from run_mcq_experiment import (
    collect_ceo_images,
    evaluate_mcq,
    compute_mcq_metrics,
)
from steering import STEERER_FILENAMES
from steering.methods import load_steerer
from vlmeval_wrapper import SteeredQwen2VLChat
from steering.hooks import SteeringHook, get_target_module


MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
STEERER_DIR = "results/Qwen3-VL-4B-Instruct/projection-mlp2"
LAYER = "projection-mlp2"
ALPHA = 1.0
TASK = "salary"
DEVICE = "cuda:0"
OUT_FILE = Path("results/ablation_results.json")


def run_mcq_once(model, ceo_images, label):
    """Run MCQ salary evaluation and return mean_race_jsd_x1000."""
    raw = evaluate_mcq(model, ceo_images, TASK, label)
    metrics = compute_mcq_metrics(raw, TASK, mode="race")
    jsd_val = metrics.get("mean_race_jsd_x1000")
    print(f"  {label}: mean_race_jsd_x1000 = {jsd_val}")
    return metrics


def install_steerer(model, steerer, alpha):
    """Remove existing hooks and install a new steerer."""
    model.remove_steering()
    hook = SteeringHook(steerer, alpha=alpha)
    target = get_target_module(model.model, LAYER)
    handle = hook.register(target)
    model.steering_hooks[LAYER] = hook
    model._hook_handles[LAYER] = handle


def main():
    results = {}

    ceo_images = collect_ceo_images("source", seed=42)
    n_ceo = len(ceo_images)
    print(f"CEO images: {n_ceo} identities")
    if n_ceo == 0:
        print("ERROR: no CEO images found in source/ceo/")
        sys.exit(1)

    # Load model once
    print(f"\nLoading model: {MODEL_PATH}")
    model = SteeredQwen2VLChat(model_path=MODEL_PATH)
    model.generate_kwargs["max_new_tokens"] = 8

    # ── BASELINE ──
    print("\n" + "=" * 60)
    print("BASELINE (unsteered)")
    print("=" * 60)
    results["baseline"] = run_mcq_once(model, ceo_images, "Baseline")

    # ── COMPONENT ABLATION (Table a) ──
    # Load steerers
    gg_steerer_path = Path(STEERER_DIR) / "geodesic_gated.pt"
    geo_svd_path = Path(STEERER_DIR) / "geo_svd.pt"
    sph_svd_path = Path(STEERER_DIR) / "sph_svd.pt"

    # 1. geo_svd
    print("\n" + "=" * 60)
    print("geo_svd (Euclidean hard proj., no gate)")
    print("=" * 60)
    steerer = load_steerer("geo_svd", str(geo_svd_path), device=DEVICE)
    install_steerer(model, steerer, ALPHA)
    results["geo_svd"] = run_mcq_once(model, ceo_images, "geo_svd")

    # 2. sph_svd
    print("\n" + "=" * 60)
    print("sph_svd (spherical hard proj., no gate)")
    print("=" * 60)
    steerer = load_steerer("sph_svd", str(sph_svd_path), device=DEVICE)
    install_steerer(model, steerer, ALPHA)
    results["sph_svd"] = run_mcq_once(model, ceo_images, "sph_svd")

    # 3. GGSS w/o gate (gate_floor=1.0 → g ≡ 1)
    print("\n" + "=" * 60)
    print("GGSS w/o gate (gate_floor=1.0)")
    print("=" * 60)
    steerer = load_steerer("geodesic_gated", str(gg_steerer_path), device=DEVICE)
    steerer.gate_floor = 1.0
    install_steerer(model, steerer, ALPHA)
    results["ggss_no_gate"] = run_mcq_once(model, ceo_images, "GGSS w/o gate")

    # 4. GGSS w/o Slerp (hard proj. + gate)
    print("\n" + "=" * 60)
    print("GGSS w/o Slerp (hard proj. + gate)")
    print("=" * 60)
    steerer = load_steerer("geodesic_gated", str(gg_steerer_path), device=DEVICE)
    steerer.ablation_no_slerp = True
    install_steerer(model, steerer, ALPHA)
    results["ggss_no_slerp"] = run_mcq_once(model, ceo_images, "GGSS w/o Slerp")

    # 5. GGSS w/o norm restore
    print("\n" + "=" * 60)
    print("GGSS w/o norm restore")
    print("=" * 60)
    steerer = load_steerer("geodesic_gated", str(gg_steerer_path), device=DEVICE)
    steerer.ablation_no_norm = True
    install_steerer(model, steerer, ALPHA)
    results["ggss_no_norm"] = run_mcq_once(model, ceo_images, "GGSS w/o norm")

    # 6. GGSS full
    print("\n" + "=" * 60)
    print("GGSS (full)")
    print("=" * 60)
    steerer = load_steerer("geodesic_gated", str(gg_steerer_path), device=DEVICE)
    install_steerer(model, steerer, ALPHA)
    results["ggss_full"] = run_mcq_once(model, ceo_images, "GGSS full")

    # ── SENSITIVITY SWEEP (Table b) ──
    kappas = [1, 2, 5, 10]
    gfloors = [0.0, 0.3, 0.5]
    results["sensitivity"] = {}

    for kappa in kappas:
        for gfloor in gfloors:
            label = f"k{kappa}_gf{gfloor}"
            print(f"\n{'=' * 60}")
            print(f"Sensitivity: kappa={kappa}, g_floor={gfloor}")
            print(f"{'=' * 60}")
            steerer = load_steerer("geodesic_gated", str(gg_steerer_path), device=DEVICE)
            steerer.kappa = kappa
            steerer.gate_floor = gfloor
            install_steerer(model, steerer, ALPHA)
            results["sensitivity"][label] = run_mcq_once(model, ceo_images, label)

    # ── SAVE ──
    # Extract key metric for easy access
    summary = {"alpha": ALPHA, "model": MODEL_PATH, "task": TASK}
    for k, v in results.items():
        if k == "sensitivity":
            summary["sensitivity"] = {}
            for sk, sv in v.items():
                summary["sensitivity"][sk] = sv.get("mean_race_jsd_x1000")
        else:
            summary[k] = v.get("mean_race_jsd_x1000")

    with open(OUT_FILE, "w") as f:
        json.dump({"summary": summary, "full": results}, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else o)

    print(f"\n{'=' * 60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to {OUT_FILE}")

    bl = summary.get("baseline", 0)
    print(f"\nBaseline:           {bl:.4f}")
    for key in ["geo_svd", "sph_svd", "ggss_no_gate", "ggss_no_slerp", "ggss_no_norm", "ggss_full"]:
        val = summary.get(key)
        if val is not None and bl > 0:
            pct = (val - bl) / bl * 100
            print(f"{key:<20s}: {val:.4f}  ({pct:+.1f}%)")

    print(f"\nSensitivity (kappa x g_floor):")
    for k in kappas:
        row = []
        for gf in gfloors:
            label = f"k{k}_gf{gf}"
            val = summary.get("sensitivity", {}).get(label, None)
            row.append(f"{val:.4f}" if val is not None else "N/A")
        print(f"  kappa={k}: " + "  ".join(row))


if __name__ == "__main__":
    main()
