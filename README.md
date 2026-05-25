# GGSS: Geodesic-Gated Spherical Steering for Debiasing Generative Vision-Language Models

This repository contains the official implementation of **GGSS** (Geodesic-Gated Spherical Steering), an inference-time debiasing method for generative Vision-Language Models (VLMs).

GGSS operates entirely at inference time by installing a lightweight forward hook on the vision-to-language projection layer. It requires **no retraining**, is **model-agnostic**, and **preserves model capability** while significantly reducing demographic bias.

## Repository Structure

```
ggss/
├── steering/                   # Core library
│   ├── __init__.py
│   ├── methods.py              # All steering methods (GGSS + baselines)
│   ├── hooks.py                # Forward hook infrastructure
│   ├── model_utils.py          # Multi-architecture VLM loading
│   └── constants.py            # Demographics, occupations, model presets
├── discovery.py                # Offline bias subspace discovery
├── vlmeval_wrapper.py          # Steered VLM inference wrapper
├── run_mcq_experiment.py       # MCQ bias evaluation (salary/education)
├── run_2afc_experiment.py      # 2AFC bias evaluation (income/education/comfort)
├── run_nurse_doctor_experiment.py  # Gender bias evaluation (occupation classification)
├── run_benchmarks.py           # VLM capability benchmarks (MMStar via VLMEvalKit)
├── run_ablation_study.py       # Component & sensitivity ablations
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** `cuml-cu12` is optional (enables GPU-accelerated logistic regression for baselines). The code gracefully falls back to scikit-learn on CPU if cuML is unavailable.

### 2. Obtain External Data

#### Counterfactual Images (FOCUS dataset)

The counterfactual face images used for bias subspace discovery and evaluation are from the **FOCUS** dataset, released as part of the REFLECT benchmark:

- **Repository:** [https://github.com/uocraW/REFLECT](https://github.com/uocraW/REFLECT)
- **Dataset path:** `focus/` directory within that repository

Download and organize the images into the following structure:

```
source/
├── cook/
│   ├── 1/
│   │   ├── Asian_man.jpg
│   │   ├── Asian_woman.jpg
│   │   ├── Black_man.jpg
│   │   ├── Black_woman.jpg
│   │   ├── Latino_man.jpg
│   │   ├── Latino_woman.jpg
│   │   ├── Middle_Eastern_man.jpg
│   │   ├── Middle_Eastern_woman.jpg
│   │   ├── White_man.jpg
│   │   └── White_woman.jpg
│   ├── 2/
│   │   └── ...
│   └── ...
├── doctor/
├── lawyer/
├── nurse/
├── teacher/
└── ceo/          # Used as held-out evaluation occupation
```

Each base identity directory contains 10 counterfactual variants (5 races × 2 genders), all matched in pose, clothing, and background.

#### VLMEvalKit (for capability benchmarks)

For running MMStar and other capability benchmarks:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && pip install -e .
```

### 3. Configure Environment

```bash
cp .env.example .env
```

No API keys are required — all inference runs locally on GPU.

### 4. Run the Full Pipeline

#### Step 1: Bias Subspace Discovery

```bash
# Discover bias subspace for Qwen3-VL-4B (race mode)
python discovery.py --preset qwen3vl --source_dir source --output_dir results

# Discover for all paper models
python discovery.py --preset pixtral --source_dir source --output_dir results
python discovery.py --preset llava16_vicuna_7b --source_dir source --output_dir results
python discovery.py --preset llava16_mistral_7b --source_dir source --output_dir results

# Gender mode (for nurse/doctor experiment)
python discovery.py --preset qwen3vl --mode gender --source_dir source --output_dir results
```

#### Step 2: Bias Evaluation

```bash
# MCQ bias (salary/education JSD)
python run_mcq_experiment.py --preset qwen3vl --source_dir source

# 2AFC bias (pairwise preference)
python run_2afc_experiment.py --preset qwen3vl --source_dir source

# Nurse/Doctor gender bias
python run_nurse_doctor_experiment.py --preset qwen3vl --mode gender --source_dir source
```

#### Step 3: Capability Evaluation

```bash
# MMStar benchmark (requires VLMEvalKit)
python run_benchmarks.py --preset qwen3vl --benchmarks MMStar
```

#### Step 4: Ablation Study

```bash
python run_ablation_study.py
```

## Method Overview

GGSS is a two-stage pipeline:

### Stage 1: Offline Counterfactual Bias Subspace Discovery

1. Collect activations from the vision-to-language projection layer using counterfactual image sets
2. Compute per-group Fréchet means on the unit hypersphere (Karcher iteration)
3. Apply Log-map to obtain tangent vectors, then SVD to extract the bias subspace V_bias
4. Calibrate the adaptive gate using the distribution of per-token bias magnitudes

### Stage 2: Online Token-Level Geodesic Steering

For each token activation at inference time:
1. Normalize to unit sphere, compute tangent vector via Log-map
2. Project onto the bias subspace to identify the biased component
3. Compute debiased target via Exp-map of the cleaned tangent vector
4. Apply adaptive gate (sigmoid of normalized bias strength) to determine steering intensity
5. Interpolate via Slerp (spherical linear interpolation) to preserve norm
6. Restore original activation magnitude

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha`   | 0.25–1.5 | Steering strength (swept per experiment) |
| `kappa`   | 5       | Gate sharpness |
| `g_floor` | 0.3     | Minimum gate value |
| `k`       | K-1 (auto) | Bias subspace dimensionality |

## Supported Models

| Model | Preset | HuggingFace Path |
|-------|--------|------------------|
| Pixtral-12B | `pixtral` | `mistral-community/pixtral-12b` |
| LLaVA-1.6-Vicuna-7B | `llava16_vicuna_7b` | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| LLaVA-1.6-Mistral-7B | `llava16_mistral_7b` | `llava-hf/llava-v1.6-mistral-7b-hf` |
| Qwen3-VL-4B-Instruct | `qwen3vl` | `Qwen/Qwen3-VL-4B-Instruct` |

The hook infrastructure (`steering/hooks.py`) supports 15+ VLM architectures via automatic layer resolution.

## Baselines Included

All baselines are implemented in `steering/methods.py`:

| Method | Variants |
|--------|----------|
| INLP | Euclidean, Spherical |
| MeanDiff (SVM-RBF) | Euclidean, Spherical |
| MeanDiff (Logistic) | Euclidean, Spherical |
| BendVLM | Euclidean, Spherical |
| SVD Ablations | Pooled Euclidean, Pooled Spherical, Per-Token Euclidean, Per-Token Spherical |

## Evaluation Metrics

| Task | Metric | Direction |
|------|--------|-----------|
| MCQ (salary/education) | Mean Race JSD × 10³ | Lower = less biased |
| 2AFC (income/education/comfort) | Race Bias Std | Lower = less biased |
| Nurse/Doctor | \|P(nurse\|man) - P(nurse\|woman)\| | Lower = less biased |
| MMStar | Accuracy | Higher = better capability |