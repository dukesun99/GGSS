"""
Bias steerers for multi-categorical VLM activation steering.

SVD variants (4):
  - Pooled  Geometric  (MultiCategoricalGeometricSteerer)
  - Pooled  Spherical  (MultiCategoricalSphericalSteerer)
  - PerToken Geometric (PerTokenGeometricSteerer)
  - PerToken Spherical (PerTokenSphericalSteerer)

Mean-difference baselines (2):
  - MeanDiffGeometricSteerer  (SVM-RBF + Euclidean mean diffs)
  - MeanDiffSphericalSteerer  (SVM-RBF + Frechet mean diffs)

INLP baselines (2):
  - INLPGeometricSteerer  (INLP subspace discovery + Euclidean projection)
  - INLPSphericalSteerer  (INLP subspace discovery + spherical projection)

BendVLM baselines (4, Gerych et al., NeurIPS 2024):
  - BendVLMPooledGeometricSteerer     (pooled + Euclidean Lagrangian)
  - BendVLMPooledSphericalSteerer     (pooled + spherical Lagrangian)
  - BendVLMPerTokenGeometricSteerer   (per-token + Euclidean Lagrangian)
  - BendVLMPerTokenSphericalSteerer   (per-token + spherical Lagrangian)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from cuml.linear_model import LogisticRegression as CumlLogisticRegression
    _HAS_CUML = True
except ImportError:
    _HAS_CUML = False

logger = logging.getLogger(__name__)

if not _HAS_CUML:
    _msg = (
        "cuML is NOT available — all LogisticRegression will run on CPU via sklearn. "
        "Install cuml-cu12 for GPU-accelerated fitting."
    )
    logger.warning(_msg)
    # Previously raised; relaxed so baseline-only runs work without cuML.
    # Training paths will fall back to sklearn (CPU).


def _make_logistic_regression(prefer_gpu: bool = True):
    """Create a LogisticRegression estimator, using cuML on GPU when available."""
    if prefer_gpu and _HAS_CUML:
        logger.info("Using cuML GPU LogisticRegression")
        return CumlLogisticRegression(max_iter=1000, verbose=0)
    if prefer_gpu and not _HAS_CUML:
        logger.warning("cuML not available, falling back to sklearn CPU LogisticRegression")
        print("[WARNING] cuML not available, falling back to sklearn CPU LogisticRegression", flush=True)
    return LogisticRegression(solver="lbfgs", max_iter=1000)


# ---------------------------------------------------------------------------
# Spherical geometry helpers
# ---------------------------------------------------------------------------

def log_map_sphere(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Logarithmic map on the unit hypersphere.

    Returns a tangent vector at *base* pointing toward *point* whose length
    equals the geodesic distance.
    """
    dot = torch.clamp(torch.dot(base, point), -1.0 + 1e-7, 1.0 - 1e-7)
    d = torch.acos(dot)
    if d.abs() < 1e-8:
        return torch.zeros_like(base)
    proj = point - dot * base
    proj = proj / (torch.norm(proj) + 1e-10)
    return proj * d


def exp_map_sphere(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Exponential map on the unit hypersphere: walk from *base* along *tangent*."""
    d = torch.norm(tangent)
    if d < 1e-8:
        return base.clone()
    direction = tangent / d
    return torch.cos(d) * base + torch.sin(d) * direction


def frechet_mean_sphere(
    points: list[torch.Tensor], max_iter: int = 100, tol: float = 1e-7
) -> torch.Tensor:
    """Iterative Frechet (Karcher) mean on the unit hypersphere."""
    mu = torch.stack(points).mean(dim=0)
    mu = mu / (torch.norm(mu) + 1e-10)

    for _ in range(max_iter):
        tangent_sum = torch.zeros_like(mu)
        for p in points:
            tangent_sum += log_map_sphere(mu, p)
        tangent_avg = tangent_sum / len(points)

        if torch.norm(tangent_avg) < tol:
            break
        mu = exp_map_sphere(mu, tangent_avg)
        mu = mu / (torch.norm(mu) + 1e-10)

    return mu


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _reshape_for_steer(h_in: torch.Tensor):
    """Flatten to [N, D] and return (h_flat, original_shape)."""
    original_shape = h_in.shape
    h = h_in.float()
    if len(original_shape) == 1:
        h = h.unsqueeze(0)
    elif len(original_shape) >= 3:
        dim = original_shape[-1]
        h = h.reshape(-1, dim)
    return h, original_shape


def _restore_shape(h: torch.Tensor, original_shape: torch.Size, dtype: torch.dtype):
    """Undo _reshape_for_steer."""
    if len(original_shape) == 1:
        h = h.squeeze(0)
    elif len(original_shape) >= 3:
        h = h.reshape(original_shape)
    return h.to(dtype)


def _discover_categories(activations_by_occ_base: dict) -> tuple[list[str], list[str]]:
    """Extract sorted race and gender lists from the first base image."""
    first_occ = next(iter(activations_by_occ_base.values()))
    sample_keys = list(first_occ[0]["acts"].keys())
    races = sorted({r for r, g in sample_keys})
    genders = sorted({g for r, g in sample_keys})
    return races, genders


# ---------------------------------------------------------------------------
# 1. Pooled Geometric (Euclidean null-space projection)
# ---------------------------------------------------------------------------

class MultiCategoricalGeometricSteerer:
    """SVD-based multi-categorical bias steerer (Euclidean null-space projection).

    Discovers a (K-1)-dimensional bias subspace via SVD on per-category
    deviations from group means across discovery occupations.

    Steering: ``h_out = h - alpha * V @ V^T @ (h - mu_global)``
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "MultiCategoricalGeometricSteerer":
        all_activations: list[torch.Tensor] = []
        shift_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].flatten().float().to(self.device)
                        race_acts[race] = act
                        all_activations.append(act)

                    if len(race_acts) < 2:
                        continue

                    center = torch.stack(list(race_acts.values())).mean(dim=0)
                    for act in race_acts.values():
                        shift_vectors.append(act - center)

        self.mu_global = torch.stack(all_activations).mean(dim=0)

        S = torch.stack(shift_vectors)
        if self.k is None:
            self.k = len(races) - 1

        _, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "geo_svd",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_shift_vectors": len(shift_vectors),
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted MultiCategoricalGeometricSteerer: k=%d, dim=%d, shifts=%d",
                     self.k, self.mu_global.shape[0], len(shift_vectors))
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device).unsqueeze(0)

        diff = h - mu
        projection = torch.matmul(diff, V)
        correction = torch.matmul(projection, V.T)
        h_steered = h - alpha * correction

        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved MultiCategoricalGeometricSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MultiCategoricalGeometricSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 2. Pooled Spherical (Riemannian null-space projection)
# ---------------------------------------------------------------------------

class MultiCategoricalSphericalSteerer:
    """SVD-based multi-categorical bias steerer using native spherical geometry.

    Discovery uses Frechet means and log maps on the unit hypersphere.
    Steering maps to tangent space at global Frechet mean, projects out
    the bias subspace, then maps back via exp map. Preserves L2 norm.
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "MultiCategoricalSphericalSteerer":
        all_unit_acts: list[torch.Tensor] = []
        shift_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].flatten().float().to(self.device)
                        act_unit = act / (torch.norm(act) + 1e-10)
                        race_acts[race] = act_unit
                        all_unit_acts.append(act_unit)

                    if len(race_acts) < 2:
                        continue

                    center = frechet_mean_sphere(list(race_acts.values()))
                    for act_unit in race_acts.values():
                        shift_vectors.append(log_map_sphere(center, act_unit))

        self.mu_global = frechet_mean_sphere(all_unit_acts).to(self.device)

        S = torch.stack(shift_vectors)
        if self.k is None:
            self.k = len(races) - 1

        _, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "sph_svd",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_shift_vectors": len(shift_vectors),
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted MultiCategoricalSphericalSteerer: k=%d, dim=%d, shifts=%d",
                     self.k, self.mu_global.shape[0], len(shift_vectors))
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        results = []

        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)

            tangent = log_map_sphere(mu, hi_unit)
            projection = torch.matmul(tangent.unsqueeze(0), V)
            correction = torch.matmul(projection, V.T).squeeze(0)
            tangent_steered = tangent - alpha * correction

            hi_steered = exp_map_sphere(mu, tangent_steered)
            hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            results.append(hi_steered * orig_norm)

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved MultiCategoricalSphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MultiCategoricalSphericalSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 2b. Paired-Average Geometric (identity-level paired differences)
# ---------------------------------------------------------------------------

class PairedAvgGeometricSteerer:
    """SVD steerer using identity-level paired-average differences.

    For each (occ, base_identity), averages the protected-attribute
    difference across all conditioning values to produce a single
    denoised difference vector.  SVD on these identity-level vectors
    captures how the protected-attribute direction varies across
    people and occupations, not across conditioning-attribute renderings
    of the same person.

    Steering is identical to MultiCategoricalGeometricSteerer.
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "PairedAvgGeometricSteerer":
        all_activations: list[torch.Tensor] = []
        diff_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                per_cond_diffs: list[torch.Tensor] = []
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].flatten().float().to(self.device)
                        race_acts[race] = act
                        all_activations.append(act)

                    if len(race_acts) < 2:
                        continue

                    stacked = torch.stack([race_acts[r] for r in sorted(race_acts)])
                    diff = stacked[0] - stacked[-1]
                    per_cond_diffs.append(diff)

                if per_cond_diffs:
                    avg_diff = torch.stack(per_cond_diffs).mean(dim=0)
                    diff_vectors.append(avg_diff)

        self.mu_global = torch.stack(all_activations).mean(dim=0)

        S = torch.stack(diff_vectors)
        if self.k is None:
            self.k = len(races) - 1

        _, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "paired_geo",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_diff_vectors": len(diff_vectors),
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted PairedAvgGeometricSteerer: k=%d, dim=%d, diffs=%d",
                     self.k, self.mu_global.shape[0], len(diff_vectors))
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)
        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device).unsqueeze(0)
        diff = h - mu
        projection = torch.matmul(diff, V)
        correction = torch.matmul(projection, V.T)
        h_steered = h - alpha * correction
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved PairedAvgGeometricSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PairedAvgGeometricSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 2c. Paired-Average Spherical (identity-level paired differences)
# ---------------------------------------------------------------------------

class PairedAvgSphericalSteerer:
    """Spherical SVD steerer using identity-level paired-average differences.

    Same paired-average logic as PairedAvgGeometricSteerer, but operates
    on the unit hypersphere via Frechet means and log/exp maps.
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "PairedAvgSphericalSteerer":
        all_unit_acts: list[torch.Tensor] = []
        diff_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                per_cond_diffs: list[torch.Tensor] = []
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].flatten().float().to(self.device)
                        act_unit = act / (torch.norm(act) + 1e-10)
                        race_acts[race] = act_unit
                        all_unit_acts.append(act_unit)

                    if len(race_acts) < 2:
                        continue

                    sorted_keys = sorted(race_acts)
                    center = frechet_mean_sphere([race_acts[r] for r in sorted_keys])
                    tangent_first = log_map_sphere(center, race_acts[sorted_keys[0]])
                    tangent_last = log_map_sphere(center, race_acts[sorted_keys[-1]])
                    per_cond_diffs.append(tangent_first - tangent_last)

                if per_cond_diffs:
                    avg_diff = torch.stack(per_cond_diffs).mean(dim=0)
                    diff_vectors.append(avg_diff)

        self.mu_global = frechet_mean_sphere(all_unit_acts).to(self.device)

        S = torch.stack(diff_vectors)
        if self.k is None:
            self.k = len(races) - 1

        _, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "paired_sph",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_diff_vectors": len(diff_vectors),
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted PairedAvgSphericalSteerer: k=%d, dim=%d, diffs=%d",
                     self.k, self.mu_global.shape[0], len(diff_vectors))
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)
        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        results = []
        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)
            tangent = log_map_sphere(mu, hi_unit)
            projection = torch.matmul(tangent.unsqueeze(0), V)
            correction = torch.matmul(projection, V.T).squeeze(0)
            tangent_steered = tangent - alpha * correction
            hi_steered = exp_map_sphere(mu, tangent_steered)
            hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            results.append(hi_steered * orig_norm)
        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved PairedAvgSphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PairedAvgSphericalSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 2d. Paired-Average Per-Token Geometric
# ---------------------------------------------------------------------------

class PairedAvgPerTokenGeometricSteerer:
    """Per-token SVD steerer using identity-level paired-average differences.

    For each (occ, base_identity), averages the per-token protected-attribute
    difference across all conditioning values.  This denoises the per-token
    signal before SVD, combining the spatial structure of per-token methods
    with the statistical robustness of paired averaging.
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "PairedAvgPerTokenGeometricSteerer":
        all_tokens: list[torch.Tensor] = []
        diff_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                per_cond_diffs: list[torch.Tensor] = []
                T_min_global = None

                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].float().to(self.device)
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        race_acts[race] = act

                    if len(race_acts) < 2:
                        continue

                    T = min(a.shape[0] for a in race_acts.values())
                    if T_min_global is None:
                        T_min_global = T
                    else:
                        T_min_global = min(T_min_global, T)

                    sorted_keys = sorted(race_acts)
                    diff = race_acts[sorted_keys[0]][:T] - race_acts[sorted_keys[-1]][:T]
                    per_cond_diffs.append(diff)

                    for act in race_acts.values():
                        all_tokens.append(act[:T])

                if per_cond_diffs and T_min_global is not None:
                    truncated = [d[:T_min_global] for d in per_cond_diffs]
                    avg_diff = torch.stack(truncated).mean(dim=0)
                    diff_vectors.append(avg_diff)

        all_tokens_cat = torch.cat(all_tokens, dim=0)
        self.mu_global = all_tokens_cat.mean(dim=0)

        all_diffs_cat = torch.cat(diff_vectors, dim=0)

        if self.k is None:
            self.k = len(races) - 1

        _, sigma, Vh = torch.linalg.svd(all_diffs_cat, full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "paired_pt_geo",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_diff_vectors": all_diffs_cat.shape[0],
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted PairedAvgPerTokenGeometricSteerer: k=%d, dim=%d, diffs=%d",
                     self.k, self.mu_global.shape[0], all_diffs_cat.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)
        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device).unsqueeze(0)
        diff = h - mu
        projection = torch.matmul(diff, V)
        correction = torch.matmul(projection, V.T)
        h_steered = h - alpha * correction
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PairedAvgPerTokenGeometricSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 2e. Paired-Average Per-Token Spherical
# ---------------------------------------------------------------------------

class PairedAvgPerTokenSphericalSteerer:
    """Per-token spherical SVD steerer using identity-level paired-average
    differences.  Combines per-token spatial structure with paired-average
    denoising, operating on the unit hypersphere.
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "PairedAvgPerTokenSphericalSteerer":
        all_unit_tokens: list[torch.Tensor] = []
        diff_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                per_cond_diffs: list[torch.Tensor] = []
                T_min_global = None

                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].float().to(self.device)
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        act_unit = act / (torch.norm(act, dim=-1, keepdim=True) + 1e-10)
                        race_acts[race] = act_unit

                    if len(race_acts) < 2:
                        continue

                    T = min(a.shape[0] for a in race_acts.values())
                    if T_min_global is None:
                        T_min_global = T
                    else:
                        T_min_global = min(T_min_global, T)

                    sorted_keys = sorted(race_acts)
                    first = race_acts[sorted_keys[0]][:T]
                    last = race_acts[sorted_keys[-1]][:T]
                    center = (first + last) / 2
                    center = center / (torch.norm(center, dim=-1, keepdim=True) + 1e-10)

                    dots_f = torch.clamp((center * first).sum(dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
                    dots_l = torch.clamp((center * last).sum(dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
                    d_f = torch.acos(dots_f)
                    d_l = torch.acos(dots_l)
                    proj_f = first - dots_f.unsqueeze(-1) * center
                    proj_l = last - dots_l.unsqueeze(-1) * center
                    pn_f = torch.norm(proj_f, dim=-1, keepdim=True) + 1e-10
                    pn_l = torch.norm(proj_l, dim=-1, keepdim=True) + 1e-10
                    tan_f = (proj_f / pn_f) * d_f.unsqueeze(-1)
                    tan_l = (proj_l / pn_l) * d_l.unsqueeze(-1)
                    mask_f = (d_f.abs() < 1e-8).unsqueeze(-1)
                    mask_l = (d_l.abs() < 1e-8).unsqueeze(-1)
                    tan_f = tan_f.masked_fill(mask_f, 0.0)
                    tan_l = tan_l.masked_fill(mask_l, 0.0)

                    per_cond_diffs.append(tan_f - tan_l)

                    for act_unit in race_acts.values():
                        all_unit_tokens.append(act_unit[:T])

                if per_cond_diffs and T_min_global is not None:
                    truncated = [d[:T_min_global] for d in per_cond_diffs]
                    avg_diff = torch.stack(truncated).mean(dim=0)
                    diff_vectors.append(avg_diff)

        all_unit_cat = torch.cat(all_unit_tokens, dim=0)
        mu = all_unit_cat.mean(dim=0)
        self.mu_global = (mu / (torch.norm(mu) + 1e-10)).to(self.device)

        all_diffs_cat = torch.cat(diff_vectors, dim=0)

        if self.k is None:
            self.k = len(races) - 1

        _, sigma, Vh = torch.linalg.svd(all_diffs_cat, full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "paired_pt_sph",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_diff_vectors": all_diffs_cat.shape[0],
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted PairedAvgPerTokenSphericalSteerer: k=%d, dim=%d, diffs=%d",
                     self.k, self.mu_global.shape[0], all_diffs_cat.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)
        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        results = []
        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)
            tangent = log_map_sphere(mu, hi_unit)
            projection = torch.matmul(tangent.unsqueeze(0), V)
            correction = torch.matmul(projection, V.T).squeeze(0)
            tangent_steered = tangent - alpha * correction
            hi_steered = exp_map_sphere(mu, tangent_steered)
            hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            results.append(hi_steered * orig_norm)
        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PairedAvgPerTokenSphericalSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 3. Per-Token Geometric
# ---------------------------------------------------------------------------

class PerTokenGeometricSteerer:
    """Per-token variant of MultiCategoricalGeometricSteerer.

    Shift vectors are computed at *each* token position independently,
    preserving spatial structure (e.g. facial vs background tokens).
    Steering formula is the same Euclidean null-space projection.
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "PerTokenGeometricSteerer":
        """Fit from per-token activations (acts tensors must be [T, D])."""
        all_tokens: list[torch.Tensor] = []
        shift_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].float().to(self.device)
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        race_acts[race] = act

                    if len(race_acts) < 2:
                        continue

                    T = min(a.shape[0] for a in race_acts.values())
                    stacked = torch.stack([a[:T] for a in race_acts.values()])
                    center = stacked.mean(dim=0)

                    for act in race_acts.values():
                        shift_vectors.append(act[:T] - center)
                        all_tokens.append(act[:T])

        all_tokens_cat = torch.cat(all_tokens, dim=0)
        all_shifts_cat = torch.cat(shift_vectors, dim=0)

        # Drop any NaN/Inf rows. Some VLMs (e.g. Idefics2/3) emit NaN at a
        # handful of vision tokens which then propagate through the SVD and
        # crash CUSOLVER (cusolverDnSgesvdj). Filtering before factorisation
        # makes the routine numerically robust without skewing the subspace.
        finite = torch.isfinite(all_shifts_cat).all(dim=1)
        if not finite.all():
            n_drop = int((~finite).sum().item())
            print(f"  [WARN] PerTokenGeometric: dropping {n_drop}/{all_shifts_cat.shape[0]} non-finite shift rows")
            all_shifts_cat = all_shifts_cat[finite]
        finite_t = torch.isfinite(all_tokens_cat).all(dim=1)
        if not finite_t.all():
            all_tokens_cat = all_tokens_cat[finite_t]

        self.mu_global = all_tokens_cat.mean(dim=0)

        if self.k is None:
            self.k = len(races) - 1

        # CUSOLVER on H100/H200 occasionally chokes on these tall-skinny
        # matrices even with NaN-free input. Move to CPU (uses LAPACK
        # gesdd/gesvd which is far more robust) then return to device.
        try:
            _, sigma, Vh = torch.linalg.svd(all_shifts_cat, full_matrices=False)
        except Exception as e:
            print(f"  [WARN] CUDA SVD failed ({type(e).__name__}); retrying on CPU.")
            shifts_cpu = all_shifts_cat.detach().cpu().float()
            _, sigma, Vh = torch.linalg.svd(shifts_cpu, full_matrices=False)
            sigma = sigma.to(all_shifts_cat.device)
            Vh = Vh.to(all_shifts_cat.device)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "per_token_geo",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_shift_vectors": all_shifts_cat.shape[0],
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted PerTokenGeometricSteerer: k=%d, dim=%d, per-token shifts=%d",
                     self.k, self.mu_global.shape[0], all_shifts_cat.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device).unsqueeze(0)

        diff = h - mu
        projection = torch.matmul(diff, V)
        correction = torch.matmul(projection, V.T)
        h_steered = h - alpha * correction

        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved PerTokenGeometricSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PerTokenGeometricSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 4. Per-Token Spherical
# ---------------------------------------------------------------------------

class PerTokenSphericalSteerer:
    """Per-token variant of MultiCategoricalSphericalSteerer.

    Uses spherical geometry at each token position independently.
    Batched computation for log maps during discovery; per-token loop
    during steering (spherical ops are not trivially batchable).
    """

    def __init__(self, k: int | None = None, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "PerTokenSphericalSteerer":
        """Fit from per-token activations using spherical geometry."""
        all_unit_tokens: list[torch.Tensor] = []
        shift_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].float().to(self.device)
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        act_unit = act / (torch.norm(act, dim=-1, keepdim=True) + 1e-10)
                        race_acts[race] = act_unit

                    if len(race_acts) < 2:
                        continue

                    T = min(a.shape[0] for a in race_acts.values())
                    stacked = torch.stack([race_acts[r][:T] for r in races if r in race_acts])
                    center = stacked.mean(dim=0)
                    center = center / (torch.norm(center, dim=-1, keepdim=True) + 1e-10)

                    # Batched log maps
                    for race in races:
                        if race not in race_acts:
                            continue
                        points = race_acts[race][:T]
                        dots = torch.clamp(
                            (center * points).sum(dim=-1), -1.0 + 1e-7, 1.0 - 1e-7
                        )
                        d = torch.acos(dots)
                        proj = points - dots.unsqueeze(-1) * center
                        proj_norms = torch.norm(proj, dim=-1, keepdim=True) + 1e-10
                        tangent_vectors = (proj / proj_norms) * d.unsqueeze(-1)
                        mask = (d.abs() < 1e-8).unsqueeze(-1)
                        tangent_vectors = tangent_vectors.masked_fill(mask, 0.0)
                        shift_vectors.append(tangent_vectors)

                    for act_unit in race_acts.values():
                        all_unit_tokens.append(act_unit[:T])

        all_unit_cat = torch.cat(all_unit_tokens, dim=0)
        mu = all_unit_cat.mean(dim=0)
        self.mu_global = (mu / (torch.norm(mu) + 1e-10)).to(self.device)

        S = torch.cat(shift_vectors, dim=0)
        # Filter NaN/Inf rows that can sneak in via vision-token activations.
        finite = torch.isfinite(S).all(dim=1)
        if not finite.all():
            n_drop = int((~finite).sum().item())
            print(f"  [WARN] PerTokenSpherical: dropping {n_drop}/{S.shape[0]} non-finite shift rows")
            S = S[finite]

        if self.k is None:
            self.k = len(races) - 1

        try:
            _, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        except (torch._C._LinAlgError, RuntimeError):
            _, sigma, Vh = torch.linalg.svd(S.cpu(), full_matrices=False)
            sigma = sigma.to(self.device)
            Vh = Vh.to(self.device)
        self.V_bias = Vh[: self.k].T.to(self.device)

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "per_token_sph",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_shift_vectors": S.shape[0],
            "dim": self.mu_global.shape[0],
        }
        logger.info("Fitted PerTokenSphericalSteerer: k=%d, dim=%d, per-token shifts=%d",
                     self.k, self.mu_global.shape[0], S.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        results = []

        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)

            tangent = log_map_sphere(mu, hi_unit)
            projection = torch.matmul(tangent.unsqueeze(0), V)
            correction = torch.matmul(projection, V.T).squeeze(0)
            tangent_steered = tangent - alpha * correction

            hi_steered = exp_map_sphere(mu, tangent_steered)
            hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            results.append(hi_steered * orig_norm)

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved PerTokenSphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PerTokenSphericalSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 5. Mean-Diff Geometric (classifier on pooled acts + Euclidean mean diffs)
# ---------------------------------------------------------------------------

class MeanDiffGeometricSteerer:
    """Baseline steerer: classifier on pooled activations + per-class Euclidean mean diffs.

    Fit: pools per-token activations to one vector per image, trains a
    classifier on pooled vectors with race labels, and computes
    delta_c = mu_c - mu_global for each class c.

    Steer: pools the input sequence, classifier predicts class c,
    then subtracts delta_c uniformly from every token.
    ``h_out = h - alpha * delta_c``

    Args:
        classifier: ``"svm_rbf"`` (default) or ``"logistic"``.
    """

    def __init__(self, classifier: str = "svm_rbf", device: str = "cuda"):
        self.clf = None
        self.classifier = classifier
        self.mu_global: torch.Tensor | None = None
        self.class_deltas: dict[str, torch.Tensor] | None = None
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def _make_clf(self):
        if self.classifier == "svm_rbf":
            return SVC(kernel="rbf")
        return _make_logistic_regression(prefer_gpu=True)

    def fit(self, activations_by_occ_base: dict) -> "MeanDiffGeometricSteerer":
        pooled_parts: list[torch.Tensor] = []
        y_parts: list[str] = []

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for (race, gender), act in base_img["acts"].items():
                    tokens = act.float().to(self.device)
                    if tokens.dim() == 1:
                        tokens = tokens.unsqueeze(0)
                    pooled_parts.append(tokens.mean(dim=0))
                    y_parts.append(race)

        X = torch.stack(pooled_parts)
        self.mu_global = X.mean(dim=0)

        classes = sorted(set(y_parts))
        self.class_deltas = {}
        for c in classes:
            mask = torch.tensor(
                [i for i, label in enumerate(y_parts) if label == c],
                dtype=torch.long, device=X.device,
            )
            mu_c = X[mask].mean(dim=0)
            self.class_deltas[c] = (mu_c - self.mu_global).cpu()

        X_np = X.cpu().numpy()
        y_np = np.array(y_parts)

        logger.info("Training %s on %d pooled samples, dim=%d",
                     self.classifier, len(y_np), X_np.shape[1])
        self.clf = self._make_clf()
        self.clf.fit(X_np, y_np)
        clf_acc = float(self.clf.score(X_np, y_np))

        self.fitted = True
        self.metadata = {
            "type": "mean_diff_geo",
            "classifier": self.classifier,
            "classes": classes,
            "clf_train_accuracy": clf_acc,
            "num_pooled_samples": len(y_np),
            "dim": int(self.mu_global.shape[0]),
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info("Fitted MeanDiffGeometricSteerer(%s): %d classes, acc=%.4f, dim=%d",
                     self.classifier, len(classes), clf_acc, self.mu_global.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        pooled = h.mean(dim=0, keepdim=True)
        c = np.asarray(self.clf.predict(pooled.detach().cpu().float().numpy())).item()
        delta = self.class_deltas[c].to(h.device)
        h_steered = h - alpha * delta.unsqueeze(0)

        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "mu_global": self.mu_global,
                "class_deltas": self.class_deltas,
                "clf_model": self.clf,
                "classifier": self.classifier,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved MeanDiffGeometricSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MeanDiffGeometricSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        classifier = data.get("classifier", "svm_rbf")
        steerer = cls(classifier=classifier, device=device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.class_deltas = {
            k: v.to(device) for k, v in data["class_deltas"].items()
        }
        steerer.clf = data.get("clf_model") or data.get("svm_model")
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 6. Mean-Diff Spherical (classifier on pooled acts + Frechet mean diffs)
# ---------------------------------------------------------------------------

class MeanDiffSphericalSteerer:
    """Baseline steerer: classifier on pooled activations + per-class Frechet mean diffs.

    Fit: pools per-token activations to one vector per image, L2-normalizes,
    trains a classifier on pooled unit vectors, computes Frechet means
    on the unit sphere, and stores delta_c = log_map(mu_global, mu_c).

    Steer: pools and normalizes the input sequence, classifier predicts
    class c, then corrects every token via tangent-space subtraction of
    delta_c and exp-map back.

    Args:
        classifier: ``"svm_rbf"`` (default) or ``"logistic"``.
    """

    def __init__(self, classifier: str = "svm_rbf", device: str = "cuda"):
        self.clf = None
        self.classifier = classifier
        self.mu_global: torch.Tensor | None = None
        self.class_deltas: dict[str, torch.Tensor] | None = None
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def _make_clf(self):
        if self.classifier == "svm_rbf":
            return SVC(kernel="rbf")
        return _make_logistic_regression(prefer_gpu=True)

    def fit(self, activations_by_occ_base: dict) -> "MeanDiffSphericalSteerer":
        pooled_parts: list[torch.Tensor] = []
        y_parts: list[str] = []

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for (race, gender), act in base_img["acts"].items():
                    tokens = act.float().to(self.device)
                    if tokens.dim() == 1:
                        tokens = tokens.unsqueeze(0)
                    pooled = tokens.mean(dim=0)
                    pooled_unit = pooled / (torch.norm(pooled) + 1e-10)
                    pooled_parts.append(pooled_unit)
                    y_parts.append(race)

        X = torch.stack(pooled_parts)

        all_unit_list = list(X.unbind(0))
        self.mu_global = frechet_mean_sphere(all_unit_list).to(self.device)

        classes = sorted(set(y_parts))
        self.class_deltas = {}
        for c in classes:
            mask = [i for i, label in enumerate(y_parts) if label == c]
            class_points = [X[i] for i in mask]
            mu_c = frechet_mean_sphere(class_points).to(self.device)
            self.class_deltas[c] = log_map_sphere(self.mu_global, mu_c).cpu()

        X_np = X.cpu().numpy()
        y_np = np.array(y_parts)

        logger.info("Training %s on %d pooled samples, dim=%d",
                     self.classifier, len(y_np), X_np.shape[1])
        self.clf = self._make_clf()
        self.clf.fit(X_np, y_np)
        clf_acc = float(self.clf.score(X_np, y_np))

        self.fitted = True
        self.metadata = {
            "type": "mean_diff_sph",
            "classifier": self.classifier,
            "classes": classes,
            "clf_train_accuracy": clf_acc,
            "num_pooled_samples": len(y_np),
            "dim": int(self.mu_global.shape[0]),
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info("Fitted MeanDiffSphericalSteerer(%s): %d classes, acc=%.4f, dim=%d",
                     self.classifier, len(classes), clf_acc, self.mu_global.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        mu = self.mu_global.to(h.device)

        pooled = h.mean(dim=0)
        pooled_unit = pooled / (torch.norm(pooled) + 1e-10)
        c = np.asarray(
            self.clf.predict(pooled_unit.detach().cpu().float().numpy().reshape(1, -1))
        ).item()
        delta = self.class_deltas[c].to(h.device)

        results = []
        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)

            tangent = log_map_sphere(mu, hi_unit)
            tangent_steered = tangent - alpha * delta

            hi_steered = exp_map_sphere(mu, tangent_steered)
            hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            results.append(hi_steered * orig_norm)

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "mu_global": self.mu_global,
                "class_deltas": self.class_deltas,
                "clf_model": self.clf,
                "classifier": self.classifier,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved MeanDiffSphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MeanDiffSphericalSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        classifier = data.get("classifier", "svm_rbf")
        steerer = cls(classifier=classifier, device=device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.class_deltas = {
            k: v.to(device) for k, v in data["class_deltas"].items()
        }
        steerer.clf = data.get("clf_model") or data.get("svm_model")
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# INLP helper (ported from SPD / Ravfogel et al., ACL 2020)
# ---------------------------------------------------------------------------

def _extract_inlp_axes(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-6,
    C: float = 1.0,
) -> tuple[list[np.ndarray], int, float]:
    """Iterative Nullspace Projection to extract discriminant axes.

    Trains a linear classifier, extracts its weight vectors as bias
    directions, projects data into the nullspace of those directions,
    and repeats until accuracy drops to random chance.

    Returns (axes, num_iterations, final_accuracy).
    """
    axes: list[np.ndarray] = []
    X_curr = X.copy()
    K = len(np.unique(y))
    stop_acc = 1.0 / K + tol
    final_acc = 0.0
    num_iter = 0

    for it in range(max_iter):
        clf = _make_logistic_regression(prefer_gpu=True)
        clf.C = C
        clf.fit(X_curr, y)
        acc = float(clf.score(X_curr, y))
        final_acc = acc
        num_iter = it + 1

        if acc <= stop_acc:
            break

        W = np.asarray(clf.coef_)  # (K, D) or (1, D) for binary
        Q, _ = np.linalg.qr(W.T)
        U = Q[:, : W.shape[0]].T  # (k, D), orthonormal rows

        for w in U:
            axes.append(w.copy())

        P = np.eye(X_curr.shape[1]) - U.T @ U
        X_curr = X_curr @ P

    return axes, num_iter, final_acc


# ---------------------------------------------------------------------------
# 7. INLP Geometric (INLP subspace + Euclidean null-space projection)
# ---------------------------------------------------------------------------

class INLPGeometricSteerer:
    """INLP-based baseline: iterative linear classifier subspace discovery
    with Euclidean null-space projection for steering.

    Fit: pools per-token activations to one vector per image, runs INLP
    on pooled vectors with race labels to discover a bias subspace V_bias,
    then computes Euclidean mu_global.

    Steer: identical to PerTokenGeometricSteerer --
    ``h_out = h - alpha * V @ V^T @ (h - mu_global)``
    """

    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-6,
        C: float = 1.0,
        device: str = "cuda",
        max_samples: int = 20_000,
    ):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k: int | None = None
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.device = device
        self.max_samples = max_samples
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "INLPGeometricSteerer":
        per_token = _HAS_CUML

        if per_token:
            X_parts: list[torch.Tensor] = []
            y_parts: list[str] = []
            for occ, base_images in activations_by_occ_base.items():
                for base_img in base_images:
                    for (race, gender), act in base_img["acts"].items():
                        tokens = act.float().to(self.device)
                        if tokens.dim() == 1:
                            tokens = tokens.unsqueeze(0)
                        X_parts.append(tokens)
                        y_parts.extend([race] * tokens.shape[0])
            X = torch.cat(X_parts, dim=0)
            if X.shape[0] > self.max_samples:
                logger.info("INLP (geo): subsampling %d -> %d rows", X.shape[0], self.max_samples)
                idx = torch.randperm(X.shape[0])[:self.max_samples]
                X = X[idx]
                y_parts = [y_parts[i] for i in idx.tolist()]
        else:
            pooled_parts: list[torch.Tensor] = []
            y_parts: list[str] = []
            for occ, base_images in activations_by_occ_base.items():
                for base_img in base_images:
                    for (race, gender), act in base_img["acts"].items():
                        tokens = act.float().to(self.device)
                        if tokens.dim() == 1:
                            tokens = tokens.unsqueeze(0)
                        pooled_parts.append(tokens.mean(dim=0))
                        y_parts.append(race)
            X = torch.stack(pooled_parts)

        self.mu_global = X.mean(dim=0)

        X_np = X.cpu().numpy()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_np = le.fit_transform(y_parts)

        level = "per-token" if per_token else "pooled"
        logger.info("Running INLP on %d %s samples, dim=%d, max_iter=%d",
                     len(y_np), level, X_np.shape[1], self.max_iter)
        axes, num_iter, final_acc = _extract_inlp_axes(
            X_np, y_np, max_iter=self.max_iter, tol=self.tol, C=self.C,
        )

        if len(axes) == 0:
            logger.warning("INLP found 0 axes; falling back to single zero vector")
            axes = [np.zeros(X_np.shape[1])]

        W = np.stack(axes)
        Q, _ = np.linalg.qr(W.T)
        V = Q[:, : W.shape[0]]  # (D, k)
        self.V_bias = torch.from_numpy(V).float().to(self.device)
        self.k = self.V_bias.shape[1]

        self.fitted = True
        self.metadata = {
            "type": "inlp_geo",
            "k": self.k,
            "inlp_iterations": num_iter,
            "inlp_final_accuracy": float(final_acc),
            "inlp_raw_axes": len(axes),
            "num_samples": len(y_np),
            "per_token": per_token,
            "num_classes": len(le.classes_),
            "classes": le.classes_.tolist(),
            "dim": int(self.mu_global.shape[0]),
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info("Fitted INLPGeometricSteerer: k=%d, %d INLP iters, final_acc=%.4f, dim=%d",
                     self.k, num_iter, final_acc, self.mu_global.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device).unsqueeze(0)

        diff = h - mu
        projection = torch.matmul(diff, V)
        correction = torch.matmul(projection, V.T)
        h_steered = h - alpha * correction

        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved INLPGeometricSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "INLPGeometricSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.k = data["k"]
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 8. INLP Spherical (INLP subspace + spherical tangent-space projection)
# ---------------------------------------------------------------------------

class INLPSphericalSteerer:
    """INLP-based baseline with spherical geometry.

    Fit: pools per-token activations to one vector per image,
    L2-normalizes, runs INLP on pooled unit-sphere activations,
    computes Frechet mean for mu_global.

    Steer: identical to PerTokenSphericalSteerer -- log-map at mu_global,
    project out V_bias, exp-map back, preserve original norm.
    """

    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-6,
        C: float = 1.0,
        device: str = "cuda",
        max_samples: int = 20_000,
    ):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k: int | None = None
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.device = device
        self.max_samples = max_samples
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "INLPSphericalSteerer":
        per_token = _HAS_CUML

        if per_token:
            X_parts: list[torch.Tensor] = []
            y_parts: list[str] = []
            for occ, base_images in activations_by_occ_base.items():
                for base_img in base_images:
                    for (race, gender), act in base_img["acts"].items():
                        tokens = act.float().to(self.device)
                        if tokens.dim() == 1:
                            tokens = tokens.unsqueeze(0)
                        norms = torch.norm(tokens, dim=-1, keepdim=True) + 1e-10
                        X_parts.append(tokens / norms)
                        y_parts.extend([race] * tokens.shape[0])
            X = torch.cat(X_parts, dim=0)
            if X.shape[0] > self.max_samples:
                logger.info("INLP (sph): subsampling %d -> %d rows", X.shape[0], self.max_samples)
                idx = torch.randperm(X.shape[0])[:self.max_samples]
                X = X[idx]
                y_parts = [y_parts[i] for i in idx.tolist()]
        else:
            pooled_parts: list[torch.Tensor] = []
            y_parts: list[str] = []
            for occ, base_images in activations_by_occ_base.items():
                for base_img in base_images:
                    for (race, gender), act in base_img["acts"].items():
                        tokens = act.float().to(self.device)
                        if tokens.dim() == 1:
                            tokens = tokens.unsqueeze(0)
                        pooled = tokens.mean(dim=0)
                        pooled_unit = pooled / (torch.norm(pooled) + 1e-10)
                        pooled_parts.append(pooled_unit)
                        y_parts.append(race)
            X = torch.stack(pooled_parts)

        all_unit_list = list(X.unbind(0))
        self.mu_global = frechet_mean_sphere(all_unit_list).to(self.device)

        X_np = X.cpu().numpy()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_np = le.fit_transform(y_parts)

        level = "per-token" if per_token else "pooled"
        logger.info("Running INLP (spherical) on %d %s samples, dim=%d, max_iter=%d",
                     len(y_np), level, X_np.shape[1], self.max_iter)
        axes, num_iter, final_acc = _extract_inlp_axes(
            X_np, y_np, max_iter=self.max_iter, tol=self.tol, C=self.C,
        )

        if len(axes) == 0:
            logger.warning("INLP found 0 axes; falling back to single zero vector")
            axes = [np.zeros(X_np.shape[1])]

        W = np.stack(axes)
        Q, _ = np.linalg.qr(W.T)
        V = Q[:, : W.shape[0]]  # (D, k)
        self.V_bias = torch.from_numpy(V).float().to(self.device)
        self.k = self.V_bias.shape[1]

        self.fitted = True
        self.metadata = {
            "type": "inlp_sph",
            "k": self.k,
            "inlp_iterations": num_iter,
            "inlp_final_accuracy": float(final_acc),
            "inlp_raw_axes": len(axes),
            "num_samples": len(y_np),
            "per_token": per_token,
            "num_classes": len(le.classes_),
            "classes": le.classes_.tolist(),
            "dim": int(self.mu_global.shape[0]),
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info("Fitted INLPSphericalSteerer: k=%d, %d INLP iters, final_acc=%.4f, dim=%d",
                     self.k, num_iter, final_acc, self.mu_global.shape[0])
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        results = []

        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)

            tangent = log_map_sphere(mu, hi_unit)
            projection = torch.matmul(tangent.unsqueeze(0), V)
            correction = torch.matmul(projection, V.T).squeeze(0)
            tangent_steered = tangent - alpha * correction

            hi_steered = exp_map_sphere(mu, tangent_steered)
            hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            results.append(hi_steered * orig_norm)

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {"V_bias": self.V_bias, "mu_global": self.mu_global,
             "k": self.k, "metadata": meta},
            path,
        )
        logger.info("Saved INLPSphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "INLPSphericalSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.k = data["k"]
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# BendVLM helpers (adapted from Gerych et al., NeurIPS 2024)
# ---------------------------------------------------------------------------

def _bendvlm_proj_matrix(embeddings: np.ndarray) -> np.ndarray:
    """SVD-based orthogonal null-space projection (BendVLM ``get_proj_matrix``).

    Computes a projection matrix that projects *out* the subspace spanned by
    ``embeddings`` (the spurious attribute directions).
    """
    from sklearn.decomposition import TruncatedSVD

    n = min(embeddings.shape[0], embeddings.shape[1])
    tSVD = TruncatedSVD(n_components=n)
    tSVD.fit_transform(embeddings)
    basis = tSVD.components_.T  # (D, n)

    proj = np.linalg.inv(basis.T @ basis)
    proj = basis @ proj @ basis.T
    proj = np.eye(proj.shape[0]) - proj
    return proj


def _bendvlm_lagrangian_solve(
    query: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    races: list[str],
    num_neighbors: int,
    proj_matrix: np.ndarray | None,
) -> np.ndarray:
    """Solve the BendVLM Lagrangian for a single query vector.

    Finds ``e*`` that maximises ``dot(e*, query)`` subject to:
      - ``||e*|| = 1``
      - ``dot(e*, proto_i) = dot(e*, proto_j)`` for all race pairs

    This makes ``e*`` equidistant (in cosine) from every race prototype.

    Returns ``e*`` as a 1-D array.
    """
    from sentence_transformers import util as st_util

    q = query.reshape(1, -1).astype(np.float64)

    if proj_matrix is not None:
        q = q @ proj_matrix.T
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-10, 1.0, norm)
        q = q / norm

    # Retrieve nearest neighbours per-race and compute prototypes
    cos_scores = st_util.cos_sim(
        torch.from_numpy(q.astype(np.float32)),
        torch.from_numpy(ref_embeddings.astype(np.float32)),
    ).numpy().reshape(-1)
    sorted_idx = np.argsort(-cos_scores)

    prototypes: list[np.ndarray] = []
    for race in races:
        race_mask = ref_labels[sorted_idx] == race
        race_indices = sorted_idx[race_mask][:num_neighbors]
        if len(race_indices) == 0:
            race_indices = sorted_idx[race_mask]
        if len(race_indices) == 0:
            prototypes.append(q.reshape(-1))
            continue
        prototypes.append(ref_embeddings[race_indices].mean(axis=0))

    # Closed-form solution.
    # Constraints  e·(pᵢ − anchor) = 0  mean e ⊥ span{pᵢ − anchor}.
    # Optimal e* = normalised projection of q onto the null space of those
    # constraint normals (maximises e·q on the unit sphere within that subspace).
    x0 = q.reshape(-1).astype(np.float64)
    anchor = prototypes[0].reshape(-1).astype(np.float64)

    if len(prototypes) <= 1:
        e_star = x0 / (np.linalg.norm(x0) + 1e-30)
        return e_star.astype(np.float32)

    M = np.column_stack(
        [p.reshape(-1).astype(np.float64) - anchor for p in prototypes[1:]]
    )  # D × (K-1)
    MtM = M.T @ M  # (K-1) × (K-1)  — tiny
    Mtq = M.T @ x0
    coeff = np.linalg.solve(MtM, Mtq)
    q_proj = x0 - M @ coeff
    norm = np.linalg.norm(q_proj)
    if norm < 1e-30:
        return x0.astype(np.float32)
    e_star = q_proj / norm
    return e_star.astype(np.float32)


def _bendvlm_lagrangian_solve_spherical(
    query: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    races: list[str],
    num_neighbors: int,
    proj_matrix: np.ndarray | None,
    mu_global: torch.Tensor,
) -> np.ndarray:
    """Spherical variant of the BendVLM Lagrangian.

    Works in the tangent space at ``mu_global``:
      1. Log-map query and prototypes into tangent space.
      2. Solve equidistance constraints in tangent space.
      3. Exp-map result back onto sphere.
    """
    from sentence_transformers import util as st_util

    q = query.reshape(1, -1).astype(np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-10:
        q = q / q_norm

    if proj_matrix is not None:
        q = q @ proj_matrix.T
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-10, 1.0, norm)
        q = q / norm

    # Retrieve neighbours and compute prototypes on the unit sphere
    cos_scores = st_util.cos_sim(
        torch.from_numpy(q.astype(np.float32)),
        torch.from_numpy(ref_embeddings.astype(np.float32)),
    ).numpy().reshape(-1)
    sorted_idx = np.argsort(-cos_scores)

    prototypes_t: list[torch.Tensor] = []
    for race in races:
        race_mask = ref_labels[sorted_idx] == race
        race_indices = sorted_idx[race_mask][:num_neighbors]
        if len(race_indices) == 0:
            race_indices = sorted_idx[race_mask]
        if len(race_indices) == 0:
            prototypes_t.append(mu_global.clone())
            continue
        pts = [torch.from_numpy(ref_embeddings[i]).float() for i in race_indices]
        proto = frechet_mean_sphere(pts)
        prototypes_t.append(proto)

    # Log-map everything into tangent space at mu_global
    mu = mu_global.float()
    q_t = torch.from_numpy(q.reshape(-1)).float()
    q_tangent = log_map_sphere(mu, q_t / (torch.norm(q_t) + 1e-10))

    proto_tangents = []
    for p in prototypes_t:
        proto_tangents.append(log_map_sphere(mu, p).numpy().astype(np.float64))

    # Solve in tangent space: equidistance in tangent norms
    x0 = q_tangent.numpy().astype(np.float64)
    anchor = proto_tangents[0]

    # Closed-form constrained-least-squares.
    # Constraint i:  ||e−pᵢ||² = ||e−anchor||²
    #   ⟹  2 e·(anchor − pᵢ) + (pᵢ·pᵢ − anchor·anchor) = 0   (affine in e)
    # Minimise ||e − x0||² subject to Ae = b  ⟹
    #   e* = x0 + Aᵀ (A Aᵀ)⁻¹ (b − A x0)
    if len(proto_tangents) <= 1:
        tangent_star = torch.from_numpy(x0.astype(np.float32))
    else:
        a_anchor = float(np.dot(anchor, anchor))
        rows = []
        bs = []
        for pt in proto_tangents[1:]:
            rows.append(2.0 * (anchor - pt))
            bs.append(a_anchor - float(np.dot(pt, pt)))
        A = np.array(rows)   # (K-1) × D
        b = np.array(bs)     # (K-1,)
        AAT = A @ A.T        # (K-1) × (K-1) — tiny
        residual = b - A @ x0
        lam = np.linalg.solve(AAT, residual)
        e_star = x0 + A.T @ lam
        tangent_star = torch.from_numpy(e_star.astype(np.float32))
    result = exp_map_sphere(mu, tangent_star)
    result = result / (torch.norm(result) + 1e-10)
    return result.numpy()


# ---------------------------------------------------------------------------
# 9. BendVLM Pooled Geometric
# ---------------------------------------------------------------------------

class BendVLMPooledGeometricSteerer:
    """BendVLM baseline (Gerych et al., NeurIPS 2024) — pooled, Euclidean.

    Adapts BendVLM's test-time debiasing to VLM hidden activations.
    Fit:  stores mean-pooled reference activations with race labels;
          builds SVD null-space projection from race-center differences.
    Steer: per-activation Lagrangian optimization to make query equidistant
           from local race prototypes (nearest-neighbor retrieval).
    """

    def __init__(self, num_neighbors: int = 10, device: str = "cuda"):
        self.num_neighbors = num_neighbors
        self.device = device
        self.ref_embeddings: np.ndarray | None = None
        self.ref_labels: np.ndarray | None = None
        self.proj_matrix: np.ndarray | None = None
        self.races: list[str] | None = None
        self.mu_global: torch.Tensor | None = None
        self.fitted = False
        self.metadata: dict[str, Any] = {}
        self._cached_delta: torch.Tensor | None = None

    def clear_cache(self):
        self._cached_delta = None

    def fit(self, activations_by_occ_base: dict) -> "BendVLMPooledGeometricSteerer":
        all_vecs: list[np.ndarray] = []
        all_labels: list[str] = []

        races, genders = _discover_categories(activations_by_occ_base)
        self.races = races

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for (race, gender), act in base_img["acts"].items():
                    vec = act.flatten().float().cpu().numpy()
                    all_vecs.append(vec)
                    all_labels.append(race)

        self.ref_embeddings = np.stack(all_vecs).astype(np.float32)
        self.ref_labels = np.array(all_labels)

        # Normalise reference embeddings
        norms = np.linalg.norm(self.ref_embeddings, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self.ref_embeddings = self.ref_embeddings / norms

        self.mu_global = torch.from_numpy(self.ref_embeddings.mean(axis=0))

        # Build projection matrix from race-center differences
        race_centers = {}
        for race in races:
            mask = self.ref_labels == race
            if mask.any():
                race_centers[race] = self.ref_embeddings[mask].mean(axis=0)

        if len(race_centers) >= 2:
            global_center = self.ref_embeddings.mean(axis=0)
            diff_vecs = []
            for race in races:
                if race in race_centers:
                    diff_vecs.append(race_centers[race] - global_center)
            center_diffs = np.stack(diff_vecs)
            # Pairwise differences for richer subspace (like BendVLM's P0_local)
            pair_diffs = []
            for i in range(len(diff_vecs)):
                for j in range(i + 1, len(diff_vecs)):
                    pair_diffs.append((diff_vecs[i] - diff_vecs[j]) / 2.0)
            if pair_diffs:
                spurious_dirs = np.concatenate([center_diffs, np.stack(pair_diffs)])
            else:
                spurious_dirs = center_diffs
            self.proj_matrix = _bendvlm_proj_matrix(spurious_dirs)
        else:
            self.proj_matrix = None

        self.fitted = True
        self.metadata = {
            "type": "bendvlm_geo",
            "num_neighbors": self.num_neighbors,
            "num_ref": len(all_vecs),
            "num_races": len(races),
            "races": races,
            "dim": self.ref_embeddings.shape[1],
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info(
            "Fitted BendVLMPooledGeometricSteerer: %d refs, %d races, dim=%d",
            len(all_vecs), len(races), self.ref_embeddings.shape[1],
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        if self._cached_delta is not None:
            delta = self._cached_delta.to(h.device)
        else:
            pooled = h.mean(dim=0).detach().cpu().float().numpy()
            pooled_norm = np.linalg.norm(pooled)

            e_star = _bendvlm_lagrangian_solve(
                pooled, self.ref_embeddings, self.ref_labels,
                self.races, self.num_neighbors, self.proj_matrix,
            )
            e_star_scaled = e_star * pooled_norm
            delta = torch.from_numpy(
                (e_star_scaled - pooled).astype(np.float32)
            ).to(h.device)
            self._cached_delta = delta

        h_steered = h + alpha * delta.unsqueeze(0)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "ref_embeddings": self.ref_embeddings,
                "ref_labels": self.ref_labels,
                "proj_matrix": self.proj_matrix,
                "races": self.races,
                "mu_global": self.mu_global,
                "num_neighbors": self.num_neighbors,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved BendVLMPooledGeometricSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "BendVLMPooledGeometricSteerer":
        data = torch.load(path, map_location="cpu", weights_only=False)
        steerer = cls(num_neighbors=data["num_neighbors"], device=device)
        steerer.ref_embeddings = data["ref_embeddings"]
        steerer.ref_labels = data["ref_labels"]
        steerer.proj_matrix = data["proj_matrix"]
        steerer.races = data["races"]
        steerer.mu_global = data["mu_global"]
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 10. BendVLM Pooled Spherical
# ---------------------------------------------------------------------------

class BendVLMPooledSphericalSteerer:
    """BendVLM baseline — pooled, spherical geometry.

    Same as BendVLMPooledGeometricSteerer but uses Fréchet means for
    prototypes and solves the Lagrangian in tangent space.
    """

    def __init__(self, num_neighbors: int = 10, device: str = "cuda"):
        self.num_neighbors = num_neighbors
        self.device = device
        self.ref_embeddings: np.ndarray | None = None
        self.ref_labels: np.ndarray | None = None
        self.proj_matrix: np.ndarray | None = None
        self.races: list[str] | None = None
        self.mu_global: torch.Tensor | None = None
        self.fitted = False
        self.metadata: dict[str, Any] = {}
        self._cached_correction: torch.Tensor | None = None

    def clear_cache(self):
        self._cached_correction = None

    def fit(self, activations_by_occ_base: dict) -> "BendVLMPooledSphericalSteerer":
        all_vecs: list[np.ndarray] = []
        all_labels: list[str] = []

        races, genders = _discover_categories(activations_by_occ_base)
        self.races = races

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for (race, gender), act in base_img["acts"].items():
                    vec = act.flatten().float().cpu().numpy()
                    all_vecs.append(vec)
                    all_labels.append(race)

        ref = np.stack(all_vecs).astype(np.float32)
        norms = np.linalg.norm(ref, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self.ref_embeddings = ref / norms
        self.ref_labels = np.array(all_labels)

        # Fréchet mean on sphere
        all_pts = [torch.from_numpy(v).float() for v in self.ref_embeddings]
        self.mu_global = frechet_mean_sphere(all_pts)

        # SVD null-space projection from race-center differences
        race_centers = {}
        for race in races:
            mask = self.ref_labels == race
            if mask.any():
                pts = [torch.from_numpy(self.ref_embeddings[i]).float()
                       for i in np.where(mask)[0]]
                race_centers[race] = frechet_mean_sphere(pts).numpy()

        if len(race_centers) >= 2:
            global_c = self.mu_global.numpy()
            diff_vecs = [race_centers[r] - global_c for r in races if r in race_centers]
            pair_diffs = []
            for i in range(len(diff_vecs)):
                for j in range(i + 1, len(diff_vecs)):
                    pair_diffs.append((diff_vecs[i] - diff_vecs[j]) / 2.0)
            spurious = np.stack(diff_vecs)
            if pair_diffs:
                spurious = np.concatenate([spurious, np.stack(pair_diffs)])
            self.proj_matrix = _bendvlm_proj_matrix(spurious)
        else:
            self.proj_matrix = None

        self.fitted = True
        self.metadata = {
            "type": "bendvlm_sph",
            "num_neighbors": self.num_neighbors,
            "num_ref": len(all_vecs),
            "num_races": len(races),
            "races": races,
            "dim": self.ref_embeddings.shape[1],
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info(
            "Fitted BendVLMPooledSphericalSteerer: %d refs, %d races, dim=%d",
            len(all_vecs), len(races), self.ref_embeddings.shape[1],
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        if self._cached_correction is not None:
            correction = self._cached_correction.to(h.device)
        else:
            pooled = h.mean(dim=0).detach().cpu().float().numpy()

            e_star_unit = _bendvlm_lagrangian_solve_spherical(
                pooled, self.ref_embeddings, self.ref_labels,
                self.races, self.num_neighbors, self.proj_matrix,
                self.mu_global,
            )

            pooled_norm = np.linalg.norm(pooled) + 1e-10
            pooled_unit = pooled / pooled_norm
            e_star_unit_np = e_star_unit.astype(np.float32)
            correction = torch.from_numpy(
                ((e_star_unit_np - pooled_unit) * pooled_norm).astype(np.float32)
            ).to(h.device)
            self._cached_correction = correction

        h_steered = h + alpha * correction.unsqueeze(0)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "ref_embeddings": self.ref_embeddings,
                "ref_labels": self.ref_labels,
                "proj_matrix": self.proj_matrix,
                "races": self.races,
                "mu_global": self.mu_global,
                "num_neighbors": self.num_neighbors,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved BendVLMPooledSphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "BendVLMPooledSphericalSteerer":
        data = torch.load(path, map_location="cpu", weights_only=False)
        steerer = cls(num_neighbors=data["num_neighbors"], device=device)
        steerer.ref_embeddings = data["ref_embeddings"]
        steerer.ref_labels = data["ref_labels"]
        steerer.proj_matrix = data["proj_matrix"]
        steerer.races = data["races"]
        steerer.mu_global = data["mu_global"]
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 11. BendVLM Per-Token Geometric
# ---------------------------------------------------------------------------

class BendVLMPerTokenGeometricSteerer:
    """BendVLM baseline — per-token, Euclidean.

    Stores all per-token reference activations with race labels.
    Steer: for each token position, retrieves nearest neighbors and solves
    the Lagrangian independently — capturing BendVLM's local bias idea
    at the token level.
    """

    def __init__(self, num_neighbors: int = 10, device: str = "cuda"):
        self.num_neighbors = num_neighbors
        self.device = device
        self.ref_embeddings: np.ndarray | None = None
        self.ref_labels: np.ndarray | None = None
        self.proj_matrix: np.ndarray | None = None
        self.races: list[str] | None = None
        self.mu_global: torch.Tensor | None = None
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "BendVLMPerTokenGeometricSteerer":
        all_tokens: list[np.ndarray] = []
        all_labels: list[str] = []

        races, genders = _discover_categories(activations_by_occ_base)
        self.races = races

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for (race, gender), act in base_img["acts"].items():
                    tokens = act.float().cpu()
                    if tokens.dim() == 1:
                        tokens = tokens.unsqueeze(0)
                    for t in range(tokens.shape[0]):
                        all_tokens.append(tokens[t].numpy())
                        all_labels.append(race)

        self.ref_embeddings = np.stack(all_tokens).astype(np.float32)
        self.ref_labels = np.array(all_labels)

        norms = np.linalg.norm(self.ref_embeddings, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self.ref_embeddings = self.ref_embeddings / norms

        self.mu_global = torch.from_numpy(self.ref_embeddings.mean(axis=0))

        # Projection matrix from per-token race-center differences
        race_centers = {}
        for race in races:
            mask = self.ref_labels == race
            if mask.any():
                race_centers[race] = self.ref_embeddings[mask].mean(axis=0)

        if len(race_centers) >= 2:
            global_center = self.ref_embeddings.mean(axis=0)
            diff_vecs = [race_centers[r] - global_center for r in races if r in race_centers]
            pair_diffs = []
            for i in range(len(diff_vecs)):
                for j in range(i + 1, len(diff_vecs)):
                    pair_diffs.append((diff_vecs[i] - diff_vecs[j]) / 2.0)
            spurious = np.stack(diff_vecs)
            if pair_diffs:
                spurious = np.concatenate([spurious, np.stack(pair_diffs)])
            self.proj_matrix = _bendvlm_proj_matrix(spurious)
        else:
            self.proj_matrix = None

        self.fitted = True
        self.metadata = {
            "type": "bendvlm_per_token_geo",
            "num_neighbors": self.num_neighbors,
            "num_ref_tokens": len(all_tokens),
            "num_races": len(races),
            "races": races,
            "dim": self.ref_embeddings.shape[1],
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info(
            "Fitted BendVLMPerTokenGeometricSteerer: %d ref tokens, %d races, dim=%d",
            len(all_tokens), len(races), self.ref_embeddings.shape[1],
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        results = []
        for i in range(h.shape[0]):
            hi = h[i].detach().cpu().float().numpy()
            orig_norm = np.linalg.norm(hi)

            e_star = _bendvlm_lagrangian_solve(
                hi, self.ref_embeddings, self.ref_labels,
                self.races, self.num_neighbors, self.proj_matrix,
            )
            e_star_scaled = e_star * orig_norm

            hi_steered = (1 - alpha) * hi + alpha * e_star_scaled
            results.append(torch.from_numpy(hi_steered).to(h.device))

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "ref_embeddings": self.ref_embeddings,
                "ref_labels": self.ref_labels,
                "proj_matrix": self.proj_matrix,
                "races": self.races,
                "mu_global": self.mu_global,
                "num_neighbors": self.num_neighbors,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved BendVLMPerTokenGeometricSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "BendVLMPerTokenGeometricSteerer":
        data = torch.load(path, map_location="cpu", weights_only=False)
        steerer = cls(num_neighbors=data["num_neighbors"], device=device)
        steerer.ref_embeddings = data["ref_embeddings"]
        steerer.ref_labels = data["ref_labels"]
        steerer.proj_matrix = data["proj_matrix"]
        steerer.races = data["races"]
        steerer.mu_global = data["mu_global"]
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# 12. BendVLM Per-Token Spherical
# ---------------------------------------------------------------------------

class BendVLMPerTokenSphericalSteerer:
    """BendVLM baseline — per-token, spherical geometry.

    Same as BendVLMPerTokenGeometricSteerer but uses Fréchet means and
    solves the Lagrangian in tangent space at the global Fréchet mean.
    """

    def __init__(self, num_neighbors: int = 10, device: str = "cuda"):
        self.num_neighbors = num_neighbors
        self.device = device
        self.ref_embeddings: np.ndarray | None = None
        self.ref_labels: np.ndarray | None = None
        self.proj_matrix: np.ndarray | None = None
        self.races: list[str] | None = None
        self.mu_global: torch.Tensor | None = None
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "BendVLMPerTokenSphericalSteerer":
        all_tokens: list[np.ndarray] = []
        all_labels: list[str] = []

        races, genders = _discover_categories(activations_by_occ_base)
        self.races = races

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for (race, gender), act in base_img["acts"].items():
                    tokens = act.float().cpu()
                    if tokens.dim() == 1:
                        tokens = tokens.unsqueeze(0)
                    for t in range(tokens.shape[0]):
                        all_tokens.append(tokens[t].numpy())
                        all_labels.append(race)

        ref = np.stack(all_tokens).astype(np.float32)
        norms = np.linalg.norm(ref, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self.ref_embeddings = ref / norms
        self.ref_labels = np.array(all_labels)

        # Fréchet mean (subsample if very large to keep fit tractable)
        n = self.ref_embeddings.shape[0]
        if n > 5000:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, 5000, replace=False)
            pts = [torch.from_numpy(self.ref_embeddings[i]).float() for i in idx]
        else:
            pts = [torch.from_numpy(v).float() for v in self.ref_embeddings]
        self.mu_global = frechet_mean_sphere(pts)

        # Projection matrix from race-center differences
        race_centers = {}
        for race in races:
            mask = self.ref_labels == race
            if mask.any():
                r_pts = [torch.from_numpy(self.ref_embeddings[i]).float()
                         for i in np.where(mask)[0][:2000]]
                race_centers[race] = frechet_mean_sphere(r_pts).numpy()

        if len(race_centers) >= 2:
            global_c = self.mu_global.numpy()
            diff_vecs = [race_centers[r] - global_c for r in races if r in race_centers]
            pair_diffs = []
            for i in range(len(diff_vecs)):
                for j in range(i + 1, len(diff_vecs)):
                    pair_diffs.append((diff_vecs[i] - diff_vecs[j]) / 2.0)
            spurious = np.stack(diff_vecs)
            if pair_diffs:
                spurious = np.concatenate([spurious, np.stack(pair_diffs)])
            self.proj_matrix = _bendvlm_proj_matrix(spurious)
        else:
            self.proj_matrix = None

        self.fitted = True
        self.metadata = {
            "type": "bendvlm_per_token_sph",
            "num_neighbors": self.num_neighbors,
            "num_ref_tokens": len(all_tokens),
            "num_races": len(races),
            "races": races,
            "dim": self.ref_embeddings.shape[1],
            "num_occupations": len(activations_by_occ_base),
        }
        logger.info(
            "Fitted BendVLMPerTokenSphericalSteerer: %d ref tokens, %d races, dim=%d",
            len(all_tokens), len(races), self.ref_embeddings.shape[1],
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        results = []
        for i in range(h.shape[0]):
            hi = h[i].detach().cpu().float()
            orig_norm = torch.norm(hi).item()
            hi_np = hi.numpy()

            e_star_unit = _bendvlm_lagrangian_solve_spherical(
                hi_np, self.ref_embeddings, self.ref_labels,
                self.races, self.num_neighbors, self.proj_matrix,
                self.mu_global,
            )
            e_star_scaled = e_star_unit * orig_norm

            hi_steered = (1 - alpha) * hi_np + alpha * e_star_scaled
            results.append(torch.from_numpy(hi_steered).to(h.device))

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "ref_embeddings": self.ref_embeddings,
                "ref_labels": self.ref_labels,
                "proj_matrix": self.proj_matrix,
                "races": self.races,
                "mu_global": self.mu_global,
                "num_neighbors": self.num_neighbors,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved BendVLMPerTokenSphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "BendVLMPerTokenSphericalSteerer":
        data = torch.load(path, map_location="cpu", weights_only=False)
        steerer = cls(num_neighbors=data["num_neighbors"], device=device)
        steerer.ref_embeddings = data["ref_embeddings"]
        steerer.ref_labels = data["ref_labels"]
        steerer.proj_matrix = data["proj_matrix"]
        steerer.races = data["races"]
        steerer.mu_global = data["mu_global"]
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# Slerp helper
# ---------------------------------------------------------------------------

def slerp(p1: torch.Tensor, p2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between unit vectors *p1* and *p2*."""
    dot = torch.clamp(torch.dot(p1, p2), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)
    if theta.abs() < 1e-8:
        return p1.clone()
    sin_theta = torch.sin(theta)
    return (torch.sin((1.0 - t) * theta) / sin_theta) * p1 + (
        torch.sin(t * theta) / sin_theta
    ) * p2


# ---------------------------------------------------------------------------
# Novel: Geodesic-Gated Spherical Steerer (pooled)
#
# Combines three ideas:
# 1. SVD bias-subspace discovery on pooled activations
# 2. Slerp (geodesic rotation) toward a per-input debiased target
# 3. vMF-inspired confidence gate: per-token adaptive alpha based on
#    how much demographic bias the activation exhibits
#
# Discovery is identical to MultiCategoricalSphericalSteerer.  The steering
# operation differs: we compute a fully debiased target via log/exp map,
# then use Slerp for the actual interpolation (following the geodesic on
# the hypersphere), with a confidence gate that suppresses steering for
# tokens that show little bias.
# ---------------------------------------------------------------------------

class GeodesicGatedSteerer:
    """Geodesic-gated spherical steerer (pooled SVD, Slerp + confidence gate).

    Fit: identical to MultiCategoricalSphericalSteerer – SVD on pooled
    spherical shift vectors to get V_bias, mu_global.  Additionally
    computes the training-set distribution of per-sample bias projection
    magnitudes for gating calibration.

    Steer: for each token
      1. log-map at mu_global -> tangent
      2. compute debiased tangent (remove V_bias projection)
      3. exp-map -> debiased target on sphere
      4. confidence gate: t = sigmoid(kappa * (bias_strength - bias_median))
         where bias_strength = ||V_bias^T @ tangent||
      5. slerp(h_unit, debiased_target, alpha * t)
      6. restore original norm

    Ablation flags (set after construction or load):
      ablation_no_slerp:  replace Slerp with hard spherical projection + gate
      ablation_no_norm:   skip norm restoration (step 6)
    """

    def __init__(self, k: int | None = None, kappa: float = 5.0,
                 gate_floor: float = 0.0, device: str = "cuda",
                 ablation_no_slerp: bool = False,
                 ablation_no_norm: bool = False):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.kappa = kappa
        self.gate_floor = gate_floor
        self.ablation_no_slerp = ablation_no_slerp
        self.ablation_no_norm = ablation_no_norm
        self.bias_median: float = 0.0
        self.bias_std: float = 1.0
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "GeodesicGatedSteerer":
        all_unit_acts: list[torch.Tensor] = []
        shift_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].flatten().float().to(self.device)
                        act_unit = act / (torch.norm(act) + 1e-10)
                        race_acts[race] = act_unit
                        all_unit_acts.append(act_unit)

                    if len(race_acts) < 2:
                        continue
                    center = frechet_mean_sphere(list(race_acts.values()))
                    for act_unit in race_acts.values():
                        shift_vectors.append(log_map_sphere(center, act_unit))

        self.mu_global = frechet_mean_sphere(all_unit_acts).to(self.device)

        S = torch.stack(shift_vectors)
        if self.k is None:
            self.k = len(races) - 1

        try:
            _, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        except (torch._C._LinAlgError, RuntimeError):
            _, sigma, Vh = torch.linalg.svd(S.cpu(), full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        # Calibrate the confidence gate from training data
        bias_strengths = []
        mu = self.mu_global
        V = self.V_bias
        for act_unit in all_unit_acts:
            tangent = log_map_sphere(mu, act_unit)
            proj = torch.matmul(tangent.unsqueeze(0), V).squeeze(0)
            bias_strengths.append(torch.norm(proj).item())
        bs = torch.tensor(bias_strengths)
        self.bias_median = float(bs.median())
        self.bias_std = float(bs.std()) + 1e-8

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "geodesic_gated",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_shift_vectors": len(shift_vectors),
            "dim": self.mu_global.shape[0],
            "kappa": self.kappa,
            "bias_median": self.bias_median,
            "bias_std": self.bias_std,
        }
        logger.info(
            "Fitted GeodesicGatedSteerer: k=%d, dim=%d, bias_median=%.4f, bias_std=%.4f",
            self.k, self.mu_global.shape[0], self.bias_median, self.bias_std,
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        gf = self.gate_floor
        no_slerp = getattr(self, "ablation_no_slerp", False)
        no_norm = getattr(self, "ablation_no_norm", False)
        results = []

        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)

            tangent = log_map_sphere(mu, hi_unit)

            proj = torch.matmul(tangent.unsqueeze(0), V).squeeze(0)
            bias_strength = torch.norm(proj)
            z = (bias_strength - self.bias_median) / self.bias_std
            raw_gate = torch.sigmoid(self.kappa * z).item()
            gate = gf + (1.0 - gf) * raw_gate

            correction = torch.matmul(proj.unsqueeze(0), V.T).squeeze(0)
            tangent_clean = tangent - correction
            target_unit = exp_map_sphere(mu, tangent_clean)
            target_unit = target_unit / (torch.norm(target_unit) + 1e-10)

            effective_t = alpha * gate
            if no_slerp:
                hi_steered = (1.0 - effective_t) * hi_unit + effective_t * target_unit
                hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            else:
                hi_steered = slerp(hi_unit, target_unit, effective_t)

            if no_norm:
                results.append(hi_steered)
            else:
                results.append(hi_steered * orig_norm)

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "V_bias": self.V_bias,
                "mu_global": self.mu_global,
                "k": self.k,
                "kappa": self.kappa,
                "gate_floor": self.gate_floor,
                "bias_median": self.bias_median,
                "bias_std": self.bias_std,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved GeodesicGatedSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "GeodesicGatedSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], kappa=data.get("kappa", 5.0),
                      gate_floor=data.get("gate_floor", 0.0), device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.bias_median = data.get("bias_median", 0.0)
        steerer.bias_std = data.get("bias_std", 1.0)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# Novel: Geodesic-Gated Per-Token Spherical Steerer
#
# Per-token variant: SVD subspace comes from per-token shift vectors
# (same as PerTokenSphericalSteerer), steering uses Slerp + confidence gate.
# ---------------------------------------------------------------------------

class GeodesicGatedPerTokenSteerer:
    """Geodesic-gated per-token spherical steerer (Slerp + confidence gate).

    Fit: identical to PerTokenSphericalSteerer – per-token SVD for V_bias,
    plus bias-strength calibration stats.

    Steer: per-token Slerp toward debiased target with confidence gating.
    """

    def __init__(self, k: int | None = None, kappa: float = 5.0,
                 gate_floor: float = 0.0, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.kappa = kappa
        self.gate_floor = gate_floor
        self.bias_median: float = 0.0
        self.bias_std: float = 1.0
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "GeodesicGatedPerTokenSteerer":
        all_unit_tokens: list[torch.Tensor] = []
        shift_vectors: list[torch.Tensor] = []

        races, genders = _discover_categories(activations_by_occ_base)

        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for gender in genders:
                    race_acts: dict[str, torch.Tensor] = {}
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].float().to(self.device)
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        act_unit = act / (torch.norm(act, dim=-1, keepdim=True) + 1e-10)
                        race_acts[race] = act_unit

                    if len(race_acts) < 2:
                        continue

                    T = min(a.shape[0] for a in race_acts.values())
                    stacked = torch.stack([race_acts[r][:T] for r in races if r in race_acts])
                    center = stacked.mean(dim=0)
                    center = center / (torch.norm(center, dim=-1, keepdim=True) + 1e-10)

                    for race in races:
                        if race not in race_acts:
                            continue
                        points = race_acts[race][:T]
                        dots = torch.clamp(
                            (center * points).sum(dim=-1), -1.0 + 1e-7, 1.0 - 1e-7
                        )
                        d = torch.acos(dots)
                        proj = points - dots.unsqueeze(-1) * center
                        proj_norms = torch.norm(proj, dim=-1, keepdim=True) + 1e-10
                        tangent_vectors = (proj / proj_norms) * d.unsqueeze(-1)
                        mask = (d.abs() < 1e-8).unsqueeze(-1)
                        tangent_vectors = tangent_vectors.masked_fill(mask, 0.0)
                        shift_vectors.append(tangent_vectors)

                    for act_unit in race_acts.values():
                        all_unit_tokens.append(act_unit[:T])

        all_unit_cat = torch.cat(all_unit_tokens, dim=0)
        mu = all_unit_cat.mean(dim=0)
        self.mu_global = (mu / (torch.norm(mu) + 1e-10)).to(self.device)

        S = torch.cat(shift_vectors, dim=0)
        if self.k is None:
            self.k = len(races) - 1

        try:
            _, sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        except (torch._C._LinAlgError, RuntimeError):
            _, sigma, Vh = torch.linalg.svd(S.cpu(), full_matrices=False)
        self.V_bias = Vh[: self.k].T.to(self.device)

        # Calibrate confidence gate from per-token training data (subsample)
        V = self.V_bias
        mu_g = self.mu_global
        n_cal = min(5000, all_unit_cat.shape[0])
        idx = torch.randperm(all_unit_cat.shape[0])[:n_cal]
        sample = all_unit_cat[idx].to(self.device)
        bias_strengths = []
        for j in range(sample.shape[0]):
            tangent = log_map_sphere(mu_g, sample[j])
            p = torch.matmul(tangent.unsqueeze(0), V).squeeze(0)
            bias_strengths.append(torch.norm(p).item())
        bs = torch.tensor(bias_strengths)
        self.bias_median = float(bs.median())
        self.bias_std = float(bs.std()) + 1e-8

        self.fitted = True
        num_base = sum(len(imgs) for imgs in activations_by_occ_base.values())
        self.metadata = {
            "type": "geodesic_gated_pt",
            "k": self.k,
            "singular_values": sigma[: self.k].tolist(),
            "explained_ratio": (sigma[: self.k] ** 2 / (sigma ** 2).sum()).tolist(),
            "num_occupations": len(activations_by_occ_base),
            "num_base_images": num_base,
            "num_races": len(races),
            "num_genders": len(genders),
            "num_shift_vectors": S.shape[0],
            "dim": self.mu_global.shape[0],
            "kappa": self.kappa,
            "bias_median": self.bias_median,
            "bias_std": self.bias_std,
        }
        logger.info(
            "Fitted GeodesicGatedPerTokenSteerer: k=%d, dim=%d, shifts=%d, "
            "bias_median=%.4f, bias_std=%.4f",
            self.k, self.mu_global.shape[0], S.shape[0],
            self.bias_median, self.bias_std,
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)

        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        gf = self.gate_floor
        results = []

        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)

            tangent = log_map_sphere(mu, hi_unit)

            proj = torch.matmul(tangent.unsqueeze(0), V).squeeze(0)
            bias_strength = torch.norm(proj)
            z = (bias_strength - self.bias_median) / self.bias_std
            raw_gate = torch.sigmoid(self.kappa * z).item()
            gate = gf + (1.0 - gf) * raw_gate

            correction = torch.matmul(proj.unsqueeze(0), V.T).squeeze(0)
            tangent_clean = tangent - correction
            target_unit = exp_map_sphere(mu, tangent_clean)
            target_unit = target_unit / (torch.norm(target_unit) + 1e-10)

            effective_t = alpha * gate
            hi_steered = slerp(hi_unit, target_unit, effective_t)
            results.append(hi_steered * orig_norm)

        h_steered = torch.stack(results)
        return _restore_shape(h_steered, orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save(
            {
                "V_bias": self.V_bias,
                "mu_global": self.mu_global,
                "k": self.k,
                "kappa": self.kappa,
                "gate_floor": self.gate_floor,
                "bias_median": self.bias_median,
                "bias_std": self.bias_std,
                "metadata": meta,
            },
            path,
        )
        logger.info("Saved GeodesicGatedPerTokenSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "GeodesicGatedPerTokenSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        steerer = cls(k=data["k"], kappa=data.get("kappa", 5.0),
                      gate_floor=data.get("gate_floor", 0.0), device=device)
        steerer.V_bias = data["V_bias"].to(device)
        steerer.mu_global = data["mu_global"].to(device)
        steerer.bias_median = data.get("bias_median", 0.0)
        steerer.bias_std = data.get("bias_std", 1.0)
        steerer.metadata = data["metadata"]
        steerer.fitted = True
        return steerer


# ---------------------------------------------------------------------------
# LEACE helpers
# ---------------------------------------------------------------------------

def _has_concept_erasure() -> bool:
    try:
        import concept_erasure  # noqa: F401
        return True
    except ImportError:
        return False


def _leace_fit_directions(
    activations_by_occ_base: dict,
    k: int,
    device: str,
    per_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit LEACE on pooled or per-token activations, return (V_bias, mu_global, meta).

    V_bias columns are the top-k eigenvectors of (I - P_leace) where P_leace
    is the LEACE projection matrix.  mu_global is the Fréchet mean on the
    unit hypersphere.
    """
    from concept_erasure import LeaceFitter

    races, genders = _discover_categories(activations_by_occ_base)
    race_to_idx = {r: i for i, r in enumerate(races)}
    n_classes = len(races)

    X_parts: list[torch.Tensor] = []
    y_indices: list[int] = []
    unit_acts: list[torch.Tensor] = []

    for occ, base_images in activations_by_occ_base.items():
        for base_img in base_images:
            for gender in genders:
                for race in races:
                    key = (race, gender)
                    if key not in base_img["acts"]:
                        continue
                    act = base_img["acts"][key].float()
                    if per_token:
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        for t in range(act.shape[0]):
                            token = act[t].to(device)
                            X_parts.append(token)
                            y_indices.append(race_to_idx[race])
                            u = token / (torch.norm(token) + 1e-10)
                            unit_acts.append(u)
                    else:
                        act_flat = act.flatten().to(device)
                        X_parts.append(act_flat)
                        y_indices.append(race_to_idx[race])
                        u = act_flat / (torch.norm(act_flat) + 1e-10)
                        unit_acts.append(u)

    X = torch.stack(X_parts)
    y_oh = torch.nn.functional.one_hot(
        torch.tensor(y_indices, dtype=torch.long), num_classes=n_classes
    ).float()

    if k is None:
        k = n_classes - 1

    dim = X.shape[1]
    fitter = LeaceFitter(dim, n_classes, dtype=torch.float32, device=device)
    fitter.update(X.to(device), y_oh.to(device))
    eraser = fitter.eraser
    P = eraser.P.to(device)  # (d, d) projection matrix

    # V_bias: eigenvectors of (I - P) for the top-k eigenvalues
    erasure_mat = torch.eye(dim, device=device) - P
    eigvals, eigvecs = torch.linalg.eigh(erasure_mat)
    # eigh returns ascending order; take top-k from the end
    V_bias = eigvecs[:, -k:].to(device)

    # Fréchet mean on unit sphere
    mu_global = frechet_mean_sphere(unit_acts).to(device)

    meta = {
        "num_samples": len(X_parts),
        "num_races": len(races),
        "num_genders": len(genders),
        "dim": dim,
        "top_eigenvalues": eigvals[-k:].tolist(),
    }
    return V_bias, mu_global, meta


# ---------------------------------------------------------------------------
# Novel: LEACE Spherical Steerer (pooled)
#
# Uses LEACE's optimal erasure directions for V_bias instead of SVD on shift
# vectors.  Steering is spherical (log-map / project / exp-map).
# ---------------------------------------------------------------------------

class LEACESphericalSteerer:
    """LEACE + Spherical steering (pooled).

    Fit: Uses LEACE to find optimal linear erasure directions for the
    protected attribute, then extracts top-k eigenvectors as V_bias.
    Steer: Standard spherical log/exp map projection.
    """

    def __init__(self, k: int = 1, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "LEACESphericalSteerer":
        V_bias, mu_global, meta = _leace_fit_directions(
            activations_by_occ_base, self.k, self.device, per_token=False
        )
        self.V_bias = V_bias
        self.mu_global = mu_global
        self.metadata = {"type": "leace_sph", "k": self.k, **meta}
        self.fitted = True
        logger.info(
            "Fitted LEACESphericalSteerer: k=%d, dim=%d, samples=%d",
            self.k, meta["dim"], meta["num_samples"],
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)
        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        results = []
        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)
            tangent = log_map_sphere(mu, hi_unit)
            projection = torch.matmul(tangent.unsqueeze(0), V)
            correction = torch.matmul(projection, V.T).squeeze(0)
            tangent_steered = tangent - alpha * correction
            hi_steered = exp_map_sphere(mu, tangent_steered)
            hi_steered = hi_steered / (torch.norm(hi_steered) + 1e-10)
            results.append(hi_steered * orig_norm)
        return _restore_shape(torch.stack(results), orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save({"V_bias": self.V_bias, "mu_global": self.mu_global,
                     "k": self.k, "metadata": meta}, path)
        logger.info("Saved LEACESphericalSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "LEACESphericalSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        s = cls(k=data["k"], device=device)
        s.V_bias = data["V_bias"].to(device)
        s.mu_global = data["mu_global"].to(device)
        s.metadata = data["metadata"]
        s.fitted = True
        return s


# ---------------------------------------------------------------------------
# Novel: LEACE Geodesic-Gated Steerer (pooled)
#
# Combines LEACE-optimal directions with Slerp + confidence gate.
# This is the most complete novel method: mathematically optimal bias
# directions (LEACE) + geodesic rotation (Slerp) + adaptive gating.
# ---------------------------------------------------------------------------

class LEACEGeodesicGatedSteerer:
    """LEACE + Slerp + confidence gate (pooled).

    Fit: LEACE directions + bias strength calibration.
    Steer: Slerp toward debiased target with confidence-gated alpha.
    """

    def __init__(self, k: int = 1, kappa: float = 5.0,
                 gate_floor: float = 0.0, device: str = "cuda"):
        self.V_bias: torch.Tensor | None = None
        self.mu_global: torch.Tensor | None = None
        self.k = k
        self.kappa = kappa
        self.gate_floor = gate_floor
        self.bias_median: float = 0.0
        self.bias_std: float = 1.0
        self.device = device
        self.fitted = False
        self.metadata: dict[str, Any] = {}

    def fit(self, activations_by_occ_base: dict) -> "LEACEGeodesicGatedSteerer":
        V_bias, mu_global, meta = _leace_fit_directions(
            activations_by_occ_base, self.k, self.device, per_token=False
        )
        self.V_bias = V_bias
        self.mu_global = mu_global

        # Calibrate gating threshold
        races, genders = _discover_categories(activations_by_occ_base)
        bias_strengths = []
        for occ, base_images in activations_by_occ_base.items():
            for base_img in base_images:
                for gender in genders:
                    for race in races:
                        key = (race, gender)
                        if key not in base_img["acts"]:
                            continue
                        act = base_img["acts"][key].flatten().float().to(self.device)
                        act_unit = act / (torch.norm(act) + 1e-10)
                        tangent = log_map_sphere(mu_global, act_unit)
                        proj = torch.matmul(tangent.unsqueeze(0), V_bias).squeeze(0)
                        bias_strengths.append(torch.norm(proj).item())
        bs = torch.tensor(bias_strengths)
        self.bias_median = float(bs.median())
        self.bias_std = float(bs.std()) + 1e-8

        self.metadata = {
            "type": "leace_geodesic_gated",
            "k": self.k,
            "kappa": self.kappa,
            "bias_median": self.bias_median,
            "bias_std": self.bias_std,
            **meta,
        }
        self.fitted = True
        logger.info(
            "Fitted LEACEGeodesicGatedSteerer: k=%d, dim=%d, "
            "bias_median=%.4f, bias_std=%.4f",
            self.k, meta["dim"], self.bias_median, self.bias_std,
        )
        return self

    def steer(self, h_in: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Not fitted")
        h, orig = _reshape_for_steer(h_in)
        V = self.V_bias.to(h.device)
        mu = self.mu_global.to(h.device)
        gf = self.gate_floor
        results = []
        for i in range(h.shape[0]):
            hi = h[i]
            orig_norm = torch.norm(hi)
            hi_unit = hi / (orig_norm + 1e-10)
            tangent = log_map_sphere(mu, hi_unit)
            proj = torch.matmul(tangent.unsqueeze(0), V).squeeze(0)
            bias_strength = torch.norm(proj)
            z = (bias_strength - self.bias_median) / self.bias_std
            raw_gate = torch.sigmoid(self.kappa * z).item()
            gate = gf + (1.0 - gf) * raw_gate
            correction = torch.matmul(proj.unsqueeze(0), V.T).squeeze(0)
            tangent_clean = tangent - correction
            target_unit = exp_map_sphere(mu, tangent_clean)
            target_unit = target_unit / (torch.norm(target_unit) + 1e-10)
            effective_t = alpha * gate
            hi_steered = slerp(hi_unit, target_unit, effective_t)
            results.append(hi_steered * orig_norm)
        return _restore_shape(torch.stack(results), orig, h_in.dtype)

    def save(self, path: str, *, model_name: str = "", layer_name: str = ""):
        meta = dict(self.metadata)
        meta["model_name"] = model_name
        meta["layer_name"] = layer_name
        torch.save({
            "V_bias": self.V_bias, "mu_global": self.mu_global,
            "k": self.k, "kappa": self.kappa,
            "gate_floor": self.gate_floor,
            "bias_median": self.bias_median, "bias_std": self.bias_std,
            "metadata": meta,
        }, path)
        logger.info("Saved LEACEGeodesicGatedSteerer to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "LEACEGeodesicGatedSteerer":
        data = torch.load(path, map_location=device, weights_only=False)
        s = cls(k=data["k"], kappa=data.get("kappa", 5.0),
                gate_floor=data.get("gate_floor", 0.0), device=device)
        s.V_bias = data["V_bias"].to(device)
        s.mu_global = data["mu_global"].to(device)
        s.bias_median = data.get("bias_median", 0.0)
        s.bias_std = data.get("bias_std", 1.0)
        s.metadata = data["metadata"]
        s.fitted = True
        return s


# ---------------------------------------------------------------------------
# Registry for dynamic loading by string name
# ---------------------------------------------------------------------------

STEERER_REGISTRY: dict[str, type] = {
    "geo_svd": MultiCategoricalGeometricSteerer,
    "sph_svd": MultiCategoricalSphericalSteerer,
    "paired_geo": PairedAvgGeometricSteerer,
    "paired_sph": PairedAvgSphericalSteerer,
    "paired_pt_geo": PairedAvgPerTokenGeometricSteerer,
    "paired_pt_sph": PairedAvgPerTokenSphericalSteerer,
    "per_token_geo": PerTokenGeometricSteerer,
    "per_token_sph": PerTokenSphericalSteerer,
    "mean_diff_geo": MeanDiffGeometricSteerer,
    "mean_diff_sph": MeanDiffSphericalSteerer,
    "mean_diff_lr_geo": MeanDiffGeometricSteerer,
    "mean_diff_lr_sph": MeanDiffSphericalSteerer,
    "inlp_geo": INLPGeometricSteerer,
    "inlp_sph": INLPSphericalSteerer,
    "bendvlm_geo": BendVLMPooledGeometricSteerer,
    "bendvlm_sph": BendVLMPooledSphericalSteerer,
    "geodesic_gated": GeodesicGatedSteerer,
    "geodesic_gated_pt": GeodesicGatedPerTokenSteerer,
    "leace_sph": LEACESphericalSteerer,
    "leace_geodesic_gated": LEACEGeodesicGatedSteerer,
}

STEERER_FILENAMES: dict[str, str] = {
    "geo_svd": "geo_svd.pt",
    "sph_svd": "sph_svd.pt",
    "paired_geo": "paired_geo.pt",
    "paired_sph": "paired_sph.pt",
    "paired_pt_geo": "paired_pt_geo.pt",
    "paired_pt_sph": "paired_pt_sph.pt",
    "per_token_geo": "per_token_geo.pt",
    "per_token_sph": "per_token_sph.pt",
    "mean_diff_geo": "mean_diff_geo.pt",
    "mean_diff_sph": "mean_diff_sph.pt",
    "mean_diff_lr_geo": "mean_diff_lr_geo.pt",
    "mean_diff_lr_sph": "mean_diff_lr_sph.pt",
    "inlp_geo": "inlp_geo.pt",
    "inlp_sph": "inlp_sph.pt",
    "bendvlm_geo": "bendvlm_geo.pt",
    "bendvlm_sph": "bendvlm_sph.pt",
    "geodesic_gated": "geodesic_gated.pt",
    "geodesic_gated_pt": "geodesic_gated_pt.pt",
    "leace_sph": "leace_sph.pt",
    "leace_geodesic_gated": "leace_geodesic_gated.pt",
}


def load_steerer(steerer_type: str, path: str, device: str = "cuda"):
    """Load a pre-fitted steerer by type name."""
    if steerer_type not in STEERER_REGISTRY:
        raise ValueError(
            f"Unknown steerer_type '{steerer_type}'. "
            f"Available: {list(STEERER_REGISTRY.keys())}"
        )
    cls = STEERER_REGISTRY[steerer_type]
    return cls.load(path, device=device)
