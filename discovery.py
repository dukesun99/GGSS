#!/usr/bin/env python3
"""
Discovery Phase: collect activations from counterfactual images and fit steerers.

Outputs are saved to ``<output_dir>/<model_short_name>/<layer>/`` so that
different models and layers can coexist.

Usage:
    python discovery.py \\
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \\
        --layers projection-mlp2 \\
        --source_dir source \\
        --output_dir results

    # Preset shorthand (Phi-3.5-vision, InternVL3-8B):
    python discovery.py --preset phi35v
    python discovery.py --preset internvl

    # Multiple layers in a single pass:
    python discovery.py --layers projection-mlp2 lm-layer10

    # Custom discovery occupations:
    python discovery.py --occupations cook doctor lawyer
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from steering import (
    RACES,
    GENDERS,
    DEFAULT_MODEL_PATH,
    DEFAULT_LAYERS,
    DEFAULT_DISCOVERY_PROMPT,
    DISCOVERY_OCCUPATIONS,
    GENDER_DISCOVERY_OCCUPATIONS,
    MODEL_PRESETS,
    ActivationCache,
    get_target_module,
    load_vlm,
    run_discovery_forward,
    MultiCategoricalGeometricSteerer,
    MultiCategoricalSphericalSteerer,
    PairedAvgGeometricSteerer,
    PairedAvgSphericalSteerer,
    PairedAvgPerTokenGeometricSteerer,
    PairedAvgPerTokenSphericalSteerer,
    PerTokenGeometricSteerer,
    PerTokenSphericalSteerer,
    MeanDiffGeometricSteerer,
    MeanDiffSphericalSteerer,
    INLPGeometricSteerer,
    INLPSphericalSteerer,
    BendVLMPooledGeometricSteerer,
    BendVLMPooledSphericalSteerer,
    GeodesicGatedSteerer,
    GeodesicGatedPerTokenSteerer,
    LEACESphericalSteerer,
    LEACEGeodesicGatedSteerer,
    STEERER_FILENAMES,
)


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_images_by_occupation(
    source_dir: str,
    occupations: list[str],
    num_per_occ: int | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Collect images organized by occupation.

    Returns:
        {occupation: [{base_id: str, variants: {(race, gender): path}}, ...]}
    """
    random.seed(seed)
    source_path = Path(source_dir)
    images_by_occ: dict[str, list[dict]] = {}

    for occ in occupations:
        occ_path = source_path / occ
        if not occ_path.exists():
            continue
        base_dirs = sorted(
            d for d in occ_path.iterdir() if d.is_dir() and d.name.isdigit()
        )
        if num_per_occ and len(base_dirs) > num_per_occ:
            base_dirs = random.sample(base_dirs, num_per_occ)

        occ_images: list[dict] = []
        for base_dir in base_dirs:
            variants: dict[tuple[str, str], str] = {}
            for race in RACES:
                for gender in GENDERS:
                    img_path = base_dir / f"{race}_{gender}.jpg"
                    if img_path.exists():
                        variants[(race, gender)] = str(img_path)
            if len(variants) == len(RACES) * len(GENDERS):
                sizes = {}
                for key, path in variants.items():
                    sizes[key] = Image.open(path).size
                unique_sizes = set(sizes.values())
                if len(unique_sizes) != 1:
                    size_summary = {f"{r}_{g}": f"{w}x{h}" for (r, g), (w, h) in sizes.items()}
                    raise ValueError(
                        f"Image size mismatch in {base_dir}: expected all variants "
                        f"to have the same dimensions but found {len(unique_sizes)} "
                        f"distinct sizes: {size_summary}"
                    )
                occ_images.append({"base_id": base_dir.name, "variants": variants})
        images_by_occ[occ] = occ_images

    return images_by_occ


# ---------------------------------------------------------------------------
# Activation collection (all layers in a single forward pass)
# ---------------------------------------------------------------------------

def collect_activations(
    model,
    processor,
    model_family: str,
    source_dir: str,
    layer_names: list[str],
    target_modules: dict[str, torch.nn.Module],
    occupations: list[str],
    prompt: str,
    num_per_occ: int | None = None,
    seed: int = 42,
    max_tokens: int = 2048,
    mode: str = "race",
) -> dict[str, tuple[dict, dict]]:
    """Collect activations from discovery images for all layers in one pass.

    Hooks are registered on every target module simultaneously so each image
    only requires a single forward pass regardless of how many layers are
    being probed.

    Args:
        model_family: Model family identifier (``"qwen2vl"``, ``"phi35v"``, ``"internvl"``).
        layer_names: Ordered list of layer name strings.
        target_modules: ``{layer_name: nn.Module}`` mapping.
        max_tokens: Cap per-token activations to this many tokens per image.
            When exceeded, a deterministic subset is sampled (consistent
            across race/gender variants of the same base image).

    Returns:
        ``{layer_name: (by_occ_base_pooled, by_occ_base_pertoken)}``
        where each dict has structure
        ``{occ: [{base_id, acts: {(race,gender): tensor}}]}``.
    """
    images_by_occ = collect_images_by_occupation(
        source_dir, occupations, num_per_occ, seed
    )

    cache = ActivationCache()
    for name in layer_names:
        cache.register_hook(target_modules[name], name)

    pooled_acc: dict[str, dict] = {ln: defaultdict(list) for ln in layer_names}
    pertoken_acc: dict[str, dict] = {ln: defaultdict(list) for ln in layer_names}

    total = sum(
        len(imgs) * len(RACES) * len(GENDERS)
        for imgs in images_by_occ.values()
    )
    pbar = tqdm(total=total, desc="Collecting activations")

    for occ, occ_images in images_by_occ.items():
        for img_info in occ_images:
            pooled_entries = {
                ln: {"base_id": img_info["base_id"], "acts": {}} for ln in layer_names
            }
            pertoken_entries = {
                ln: {"base_id": img_info["base_id"], "acts": {}} for ln in layer_names
            }

            # Pre-compute a deterministic token-subsample index for this
            # base image so every race/gender variant uses the same subset.
            subsample_indices: dict[str, torch.Tensor | None] = {}

            for (race, gender), img_path in img_info["variants"].items():
                image = Image.open(img_path).convert("RGB")

                with torch.no_grad():
                    run_discovery_forward(
                        model, processor, model_family,
                        image, prompt, model.device,
                    )

                for ln in layer_names:
                    act = cache.activations[ln]

                    # Flatten to [T, D] regardless of how many leading dims
                    D = act.shape[-1]
                    act_flat = act.reshape(-1, D).detach().cpu()

                    act_pooled = act_flat.mean(dim=0).detach()

                    # Subsample tokens if needed (same indices for all variants)
                    T = act_flat.shape[0]
                    if T > max_tokens:
                        if ln not in subsample_indices:
                            gen = torch.Generator().manual_seed(seed + hash(img_info["base_id"]))
                            subsample_indices[ln] = torch.randperm(T, generator=gen)[:max_tokens].sort().values
                        act_flat = act_flat[subsample_indices[ln]]

                    # In gender mode, swap the key order so position 0
                    # becomes the protected attribute (gender) and
                    # position 1 becomes the conditioning (race).
                    act_key = (gender, race) if mode == "gender" else (race, gender)
                    pooled_entries[ln]["acts"][act_key] = act_pooled
                    pertoken_entries[ln]["acts"][act_key] = act_flat

                cache.clear()
                pbar.update(1)

            for ln in layer_names:
                pooled_acc[ln][occ].append(pooled_entries[ln])
                pertoken_acc[ln][occ].append(pertoken_entries[ln])

    pbar.close()
    cache.remove_hooks()

    result: dict[str, tuple[dict, dict]] = {}
    for ln in layer_names:
        total_collected = sum(
            len(entry["acts"])
            for entries in pooled_acc[ln].values()
            for entry in entries
        )
        print(f"  [{ln}] Collected {total_collected} activations across {len(images_by_occ)} occupations")
        result[ln] = (dict(pooled_acc[ln]), dict(pertoken_acc[ln]))

    return result


# ---------------------------------------------------------------------------
# Steerer fitting and saving
# ---------------------------------------------------------------------------

def _model_short_name(model_path: str) -> str:
    """Extract a directory-safe short name from a HF model path."""
    return model_path.rstrip("/").split("/")[-1]


def _subsample_pertoken(
    by_occ_base_pertoken: dict,
    frac: float,
    seed: int = 42,
) -> dict:
    """Subsample per-token activations to *frac* of original tokens.

    For each base image the same random token indices are used across all
    (race, gender) variants so that every method sees an identical subset.
    """
    if frac >= 1.0:
        return by_occ_base_pertoken

    total_before = 0
    total_after = 0

    for occ, base_images in by_occ_base_pertoken.items():
        for base_img in base_images:
            acts = base_img["acts"]
            if not acts:
                continue

            T_min = min(a.shape[0] for a in acts.values())
            total_before += T_min * len(acts)
            keep = max(1, int(T_min * frac))

            gen = torch.Generator().manual_seed(seed + hash(base_img.get("base_id", id(base_img))))
            idx = torch.randperm(T_min, generator=gen)[:keep].sort().values

            for key in acts:
                acts[key] = acts[key][idx]

            total_after += keep * len(acts)

    print(f"  Per-token subsample: {frac:.0%} -> {total_after}/{total_before} token-vectors kept")
    return by_occ_base_pertoken


def fit_and_save_all(
    activations_by_layer: dict[str, tuple[dict, dict]],
    output_dir: str,
    model_path: str,
    pertoken_frac: float = 1.0,
    seed: int = 42,
    mode: str = "race",
    n_components: int | None = None,
):
    """Fit all steerers for each layer and save them to disk.

    Args:
        n_components: Number of SVD components for SVD-based steerers.
            None = auto (K-1). Only affects the 4 SVD steerer types.
    """
    short = _model_short_name(model_path)

    for layer_name, (by_occ_base_pooled, by_occ_base_pertoken) in activations_by_layer.items():
        if pertoken_frac < 1.0:
            _subsample_pertoken(by_occ_base_pertoken, pertoken_frac, seed=seed)

        layer_tag = layer_name if mode == "race" else f"{layer_name}_{mode}"
        out = Path(output_dir) / short / layer_tag
        out.mkdir(parents=True, exist_ok=True)

        save_kw = {"model_name": model_path, "layer_name": layer_name}

        k_label = f" (k={n_components})" if n_components is not None else " (k=auto)"

        print(f"\n{'=' * 50}")
        print(f"Fitting steerers for layer: {layer_name}{k_label}")
        print(f"{'=' * 50}")

        def _fit_save(steerer, data, key, label):
            dest = out / STEERER_FILENAMES[key]
            if dest.exists():
                print(f"\n--- {label} --- SKIPPED (exists: {dest.name})")
                return
            print(f"\n--- {label} ---")
            # Resilient: a single fit failure shouldn't stop the rest.
            # We log the traceback and continue so other methods can still
            # be fit (important because some failures, e.g. CUSOLVER NaN
            # crashes on per-token SVD, are method-specific).
            try:
                steerer.fit(data)
                steerer.save(str(dest), **save_kw)
            except Exception as e:
                import traceback
                print(f"  [WARN] Fitting {label} failed: {type(e).__name__}: {e}")
                print("  [WARN] continuing with remaining methods…")
                traceback.print_exc()

        _fit_save(MultiCategoricalGeometricSteerer(k=n_components), by_occ_base_pooled, "geo_svd", "Fitting Pooled Geometric SVD")
        _fit_save(MultiCategoricalSphericalSteerer(k=n_components), by_occ_base_pooled, "sph_svd", "Fitting Pooled Spherical SVD")
        _fit_save(PairedAvgGeometricSteerer(k=n_components), by_occ_base_pooled, "paired_geo", "Fitting Paired-Avg Geometric SVD")
        _fit_save(PairedAvgSphericalSteerer(k=n_components), by_occ_base_pooled, "paired_sph", "Fitting Paired-Avg Spherical SVD")
        _fit_save(PairedAvgPerTokenGeometricSteerer(k=n_components), by_occ_base_pertoken, "paired_pt_geo", "Fitting Paired-Avg Per-Token Geometric SVD")
        _fit_save(PairedAvgPerTokenSphericalSteerer(k=n_components), by_occ_base_pertoken, "paired_pt_sph", "Fitting Paired-Avg Per-Token Spherical SVD")
        _fit_save(PerTokenGeometricSteerer(k=n_components), by_occ_base_pertoken, "per_token_geo", "Fitting Per-Token Geometric SVD")
        _fit_save(PerTokenSphericalSteerer(k=n_components), by_occ_base_pertoken, "per_token_sph", "Fitting Per-Token Spherical SVD")
        _fit_save(MeanDiffGeometricSteerer(classifier="svm_rbf"), by_occ_base_pertoken, "mean_diff_geo", "Fitting Mean-Diff Geometric (SVM-RBF)")
        _fit_save(MeanDiffSphericalSteerer(classifier="svm_rbf"), by_occ_base_pertoken, "mean_diff_sph", "Fitting Mean-Diff Spherical (SVM-RBF)")
        _fit_save(MeanDiffGeometricSteerer(classifier="logistic"), by_occ_base_pertoken, "mean_diff_lr_geo", "Fitting Mean-Diff Geometric (Logistic)")
        _fit_save(MeanDiffSphericalSteerer(classifier="logistic"), by_occ_base_pertoken, "mean_diff_lr_sph", "Fitting Mean-Diff Spherical (Logistic)")
        _fit_save(INLPGeometricSteerer(), by_occ_base_pertoken, "inlp_geo", "Fitting INLP Geometric")
        _fit_save(INLPSphericalSteerer(), by_occ_base_pertoken, "inlp_sph", "Fitting INLP Spherical")
        _fit_save(BendVLMPooledGeometricSteerer(), by_occ_base_pooled, "bendvlm_geo", "Fitting BendVLM Pooled Geometric")
        _fit_save(BendVLMPooledSphericalSteerer(), by_occ_base_pooled, "bendvlm_sph", "Fitting BendVLM Pooled Spherical")

        # --- Novel methods: Geodesic-Gated (Slerp + confidence gate) ---
        _fit_save(GeodesicGatedSteerer(k=n_components), by_occ_base_pooled, "geodesic_gated", "Fitting Geodesic-Gated (pooled)")
        _fit_save(GeodesicGatedPerTokenSteerer(k=n_components), by_occ_base_pertoken, "geodesic_gated_pt", "Fitting Geodesic-Gated Per-Token")

        # --- LEACE-based methods ---
        try:
            _fit_save(LEACESphericalSteerer(k=n_components), by_occ_base_pooled, "leace_sph", "Fitting LEACE Spherical")
            _fit_save(LEACEGeodesicGatedSteerer(k=n_components), by_occ_base_pooled, "leace_geodesic_gated", "Fitting LEACE Geodesic-Gated")
        except ImportError:
            print("  [SKIP] concept-erasure not installed, skipping LEACE steerers")

        print(f"\nAll steerers saved to {out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discovery phase: collect activations and fit steerers"
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
        help=(
            "Target layer(s) to discover on. Multiple layers are probed in a "
            "single forward pass. Examples: projection-mlp2, lm-layer10, "
            "vision-block20"
        ),
    )
    parser.add_argument("--source_dir", default="source")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--prompt", default=DEFAULT_DISCOVERY_PROMPT)
    parser.add_argument(
        "--occupations",
        nargs="+",
        default=DISCOVERY_OCCUPATIONS,
        help=f"Discovery occupations (default: {DISCOVERY_OCCUPATIONS})",
    )
    parser.add_argument(
        "--num_per_occ",
        type=int,
        default=None,
        help="Max base images per occupation (None = all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max per-token activations per image (subsampled if exceeded)",
    )
    parser.add_argument(
        "--pertoken_frac",
        type=float,
        default=1.0,
        help=(
            "Fraction of per-token activations to keep for fitting "
            "(0 < frac <= 1.0). Use < 1.0 for models that produce very "
            "large token counts (e.g. InternVL). Subsampling is applied "
            "uniformly to all per-token methods for a fair comparison."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["race", "gender"],
        default="race",
        help=(
            "Debiasing mode. 'race' discovers a racial-bias subspace "
            "(K=5 race categories, conditioned on gender). 'gender' "
            "discovers a gender-bias subspace (K=2 gender categories, "
            "conditioned on race). Default: race."
        ),
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help=(
            "Number of SVD components for the bias subspace. "
            "Default: K-1 (auto from protected-attribute cardinality). "
            "Set higher (e.g. 4) to capture richer subspaces even when "
            "K is small (e.g. gender K=2). Only affects SVD-based steerers."
        ),
    )
    args = parser.parse_args()

    if args.preset:
        cfg = MODEL_PRESETS[args.preset]
        args.model_path = cfg["model_path"]
        args.layers = cfg["layers"]

    # In gender mode, default to occupations that exclude nurse/doctor
    # (held out for evaluation) unless explicitly overridden.
    if args.mode == "gender" and args.occupations == DISCOVERY_OCCUPATIONS:
        args.occupations = GENDER_DISCOVERY_OCCUPATIONS

    print("=" * 60)
    print("DISCOVERY PHASE")
    print("=" * 60)
    print(f"Model:         {args.model_path}")
    print(f"Layers:        {args.layers}")
    print(f"Mode:          {args.mode}")
    print(f"Source:        {args.source_dir}")
    print(f"Occupations:   {args.occupations}")
    print(f"Output:        {args.output_dir}")
    if args.pertoken_frac < 1.0:
        print(f"Per-token frac: {args.pertoken_frac:.0%}")
    if args.mode == "gender":
        print(f"  Categories (protected): {GENDERS}  (K={len(GENDERS)})")
        print(f"  Conditioning:           {RACES}")
    else:
        print(f"  Categories (protected): {RACES}  (K={len(RACES)})")
        print(f"  Conditioning:           {GENDERS}")
    n_comp_label = args.n_components if args.n_components is not None else "auto (K-1)"
    print(f"  SVD components:         {n_comp_label}")

    # Load model (auto-detects family)
    print(f"\nLoading model: {args.model_path}")
    model, processor, model_family = load_vlm(args.model_path)
    print(f"  Detected family: {model_family}")

    # Resolve all target modules
    target_modules: dict[str, torch.nn.Module] = {}
    for layer_name in args.layers:
        target_modules[layer_name] = get_target_module(model, layer_name)
        print(f"  Resolved layer '{layer_name}' -> {type(target_modules[layer_name]).__name__}")

    # Collect activations (all layers in one pass)
    print("\nCollecting activations...")
    activations_by_layer = collect_activations(
        model,
        processor,
        model_family,
        args.source_dir,
        args.layers,
        target_modules,
        args.occupations,
        args.prompt,
        num_per_occ=args.num_per_occ,
        seed=args.seed,
        max_tokens=args.max_tokens,
        mode=args.mode,
    )

    # Fit and save
    fit_and_save_all(
        activations_by_layer,
        args.output_dir,
        args.model_path,
        pertoken_frac=args.pertoken_frac,
        seed=args.seed,
        mode=args.mode,
        n_components=args.n_components,
    )

    print("\nDiscovery complete.")


if __name__ == "__main__":
    main()
