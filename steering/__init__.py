"""Activation steering library for VLM bias mitigation."""

from .constants import (
    RACES,
    GENDERS,
    DEFAULT_MODEL_PATH,
    DEFAULT_LAYER,
    DEFAULT_LAYERS,
    DEFAULT_DISCOVERY_PROMPT,
    DISCOVERY_OCCUPATIONS,
    GENDER_DISCOVERY_OCCUPATIONS,
    GENDER_EVAL_OCCUPATIONS,
    MODEL_PRESETS,
)
from .hooks import ActivationCache, SteeringHook, get_target_module
from .methods import (
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
    STEERER_REGISTRY,
    STEERER_FILENAMES,
    load_steerer,
)
from .model_utils import (
    detect_model_family,
    load_vlm,
    load_image_internvl,
    prepare_discovery_inputs,
    prepare_generate_inputs,
    run_discovery_forward,
)
