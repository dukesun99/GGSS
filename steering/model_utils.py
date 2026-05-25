"""Model-family abstraction for VLM loading and input preparation.

Supports:
  - **qwen2vl**: Qwen2.5-VL family (uses ``qwen_vl_utils`` for vision processing)
    Layout: ``model.model.visual`` (.blocks, .merger.mlp) / ``model.model`` (LM .layers)
  - **qwen3vl**: Qwen3-VL family (uses ``processor.apply_chat_template`` for inputs)
    Layout: ``model.model.visual`` (.blocks, .merger.linear_fc2) / ``model.model.language_model`` (LM .layers)
  - **qwen35**: Qwen3.5 family (hybrid GatedDeltaNet + MoE, native multimodal)
    Layout: ``model.model.visual`` (.blocks, .merger.linear_fc2) / ``model.model.language_model`` (LM .layers)
  - **phi35v**: Phi-3.5-vision family (uses HF ``AutoModelForCausalLM``)
    Layout: ``model.model.vision_embed_tokens`` (.img_projection) / ``model.model.layers``
  - **internvl**: InternVL3 family (uses HF ``AutoModel`` + custom code)
    Layout: ``model.vision_model`` / ``model.mlp1`` / ``model.language_model``
  - **llama32v**: Llama 3.2 Vision family (cross-attention adapter architecture)
    Layout: ``model.vision_model`` / ``model.multi_modal_projector`` / ``model.language_model``
  - **gemma4**: Gemma 4 family (dense multimodal with vision + audio)
    Layout: ``model.model.vision_tower`` / ``model.model.embed_vision`` / ``model.model.language_model``

Layer name mapping (``projection-mlp2``, ``lm-layerN``, ``vision-blockN``) is
handled by :func:`~steering.hooks.get_target_module` for all families.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

_FAMILY_PATTERNS: dict[str, list[str]] = {
    "qwen35": ["qwen3.5"],
    "qwen3vl": ["qwen3-vl"],
    "qwen2vl": ["qwen2.5-vl", "qwen2-vl"],
    "phi35v": ["phi-3.5-vision", "phi-3-vision"],
    "internvl": ["internvl"],
    "llama32v": ["llama-3.2", "llama3.2"],
    "gemma4": ["gemma-4", "gemma4"],
    "gemma3": ["gemma-3", "gemma3"],
    "idefics3": ["idefics3", "smolvlm"],
    "idefics2": ["idefics2"],
    "pixtral": ["pixtral"],
    "phi4mm": ["phi-4-multimodal", "phi4mm"],
    "bunny": ["bunny-v1", "bunny_llama3", "bunny-llama"],
    "minicpm_v45": ["minicpm-v-4_5", "minicpm_v45", "minicpm-v-4.5"],
    "minicpm_l3": ["minicpm-llama3-v", "minicpm_l3", "minicpm-v-2_5"],
    "glm41v": ["glm-4.1v", "glm41v"],
    "glm4v": ["glm-4v", "glm4v"],
    # LLaVA-1.5 family (must come BEFORE llava_next pattern, since neither
    # "llava-1.5" nor "llava15" matches "llava-v1.6"; both are independent.)
    "llava_onevision": ["llava-onevision", "llava_onevision"],
    "video_llava": ["video-llava", "video_llava"],
    "vip_llava": ["vip-llava", "vip_llava"],
    "llava15": ["llava-1.5", "llava15", "mantis-8b-siglip-llama3", "mantis_siglip"],
    # Granite-Vision ships as LlavaNextForConditionalGeneration, so it
    # belongs in the llava_next family.
    "varco_vision": ["varco-vision"],
    "llava_next_video": ["llava-next-video", "llavanext-video"],
    "llava_next": ["llava-v1.6", "llava-next", "llavanext", "granite-vision", "falcon-11b-vlm", "falcon_vlm"],
    "kimi_vl": ["kimi-vl", "kimi_vl"],
    "deepseek_vl2": ["deepseek-vl2", "deepseek_vl2"],
    "ovis": ["ovis", "ovis2", "ovis1.6"],
    "molmo2": ["molmo2"],
    "molmo": ["molmo"],
    "aya_vision": ["aya-vision", "aya_vision"],
    "fastvlm": ["fastvlm"],
    "nemotron_vl": ["nemotron", "nemotron-nano"],
}


def detect_model_family(model_path: str) -> str:
    """Auto-detect the model family from a HuggingFace model path."""
    lower = model_path.lower().replace(" ", "")
    for family, patterns in _FAMILY_PATTERNS.items():
        if any(p in lower for p in patterns):
            return family
    raise ValueError(
        f"Cannot auto-detect model family for '{model_path}'. "
        f"Supported families: {list(_FAMILY_PATTERNS.keys())}"
    )


# ---------------------------------------------------------------------------
# InternVL image preprocessing
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_internvl_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def _dynamic_preprocess_internvl(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    w, h = image.size
    ratios = sorted(
        set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        ),
        key=lambda x: x[0] * x[1],
    )
    target = _find_closest_aspect_ratio(w / h, ratios, w, h, image_size)
    tw, th = image_size * target[0], image_size * target[1]
    resized = image.resize((tw, th))
    blocks = target[0] * target[1]
    imgs = []
    for i in range(blocks):
        box = (
            (i % (tw // image_size)) * image_size,
            (i // (tw // image_size)) * image_size,
            ((i % (tw // image_size)) + 1) * image_size,
            ((i // (tw // image_size)) + 1) * image_size,
        )
        imgs.append(resized.crop(box))
    if use_thumbnail and len(imgs) != 1:
        imgs.append(image.resize((image_size, image_size)))
    return imgs


def load_image_internvl(image, input_size=448, max_num=12):
    """Preprocess a PIL image (or file path) for InternVL.

    Returns a ``[num_tiles, C, H, W]`` pixel-values tensor (float32).
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    transform = _build_internvl_transform(input_size)
    tiles = _dynamic_preprocess_internvl(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num,
    )
    return torch.stack([transform(t) for t in tiles])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_vlm(
    model_path: str,
    torch_dtype=torch.bfloat16,
    device_map: str = "auto",
) -> tuple[Any, Any, str]:
    """Load a VLM model and processor, auto-detecting the family.

    Returns:
        ``(model, processor_or_tokenizer, model_family)``

    For InternVL the second return value is a tokenizer (not a processor).
    """
    family = detect_model_family(model_path)

    if family == "qwen35":
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "qwen2vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "phi35v":
        from transformers import AutoModelForCausalLM, DynamicCache

        # Phi-3.5-vision's custom code calls DynamicCache methods
        # that were removed in newer transformers versions.
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, new_seq_len, layer_idx=0: (
                self.get_seq_length(layer_idx)
            )
        if not isinstance(getattr(DynamicCache, "seen_tokens", None), property):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "to_legacy_cache"):
            def _to_legacy(self):
                return [(layer.keys, layer.values) for layer in self.layers]
            DynamicCache.to_legacy_cache = _to_legacy
        if not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod  # type: ignore[misc]
            def _from_legacy(cls, past_key_values=None, **kwargs):
                cache = cls()
                if past_key_values is not None:
                    for i, (k, v) in enumerate(past_key_values):
                        cache.update(k, v, i)
                return cache
            DynamicCache.from_legacy_cache = _from_legacy

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=True,
            torch_dtype="auto", _attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, num_crops=4,
        )

    elif family == "llama32v":
        from transformers import MllamaForConditionalGeneration

        model = MllamaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "gemma4":
        from transformers import Gemma4ForConditionalGeneration

        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "gemma3":
        # Gemma 3 vision models (e.g. google/gemma-3-4b-it) use
        # Gemma3ForConditionalGeneration and the SigLIP vision tower with a
        # simple MLP multi-modal projector.
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "kimi_vl":
        # Moonshot Kimi-VL ships its own modeling file. It was authored against
        # transformers<=4.45 and imports helpers removed in newer releases.
        # We monkey-patch the missing symbols into transformers before the
        # custom code is imported by ``trust_remote_code=True``.
        import transformers.utils.import_utils as _tfx

        if not hasattr(_tfx, "is_torch_fx_available"):
            try:
                from importlib import import_module
                import_module("torch.fx")
                _tfx.is_torch_fx_available = lambda: True
            except Exception:  # pragma: no cover
                _tfx.is_torch_fx_available = lambda: False

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )

    elif family == "deepseek_vl2":
        # DeepSeek-VL2 uses a custom DeepseekVLV2ForCausalLM class and a
        # custom VLChatProcessor. We fall back to trust_remote_code and
        # handle input preparation below.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        try:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True,
            )
        except Exception:  # noqa: BLE001
            # DeepSeek-VL2 exposes processing via deepseek_vl2.VLChatProcessor
            from importlib import import_module
            try:
                vl2 = import_module("deepseek_vl2.models")
                processor = vl2.VLChatProcessor.from_pretrained(model_path)
            except ModuleNotFoundError as e:  # pragma: no cover
                raise ImportError(
                    "DeepSeek-VL2 requires the `deepseek-vl2` package "
                    "(pip install deepseek-vl2). Original error: %s" % e,
                ) from e

    elif family == "molmo":
        # Relax the dynamic-module import checker so a legacy "import
        # tensorflow" in Molmo's preprocessor does not abort loading.
        import transformers.dynamic_module_utils as _dmu
        if not getattr(_dmu, "_molmo_imports_patched", False):
            _orig_check = _dmu.check_imports
            def _lenient_check(filename):
                try:
                    return _orig_check(filename)
                except ImportError as e:
                    if "tensorflow" in str(e).lower():
                        return []
                    raise
            _dmu.check_imports = _lenient_check
            _dmu._molmo_imports_patched = True

        # AllenAI Molmo's remote code predates the ``all_tied_weights_keys``
        # API introduced in transformers>=5. Patch ``PreTrainedModel``
        # helpers to gracefully fall back before loading.
        import transformers.modeling_utils as _tmu
        if not getattr(_tmu.PreTrainedModel, "_molmo_patched", False):
            _orig_mark = _tmu.PreTrainedModel.mark_tied_weights_as_initialized
            def _safe_mark_tied(self, *a, **kw):
                if not hasattr(self, "all_tied_weights_keys"):
                    tied = getattr(self, "_tied_weights_keys", None) or []
                    try:
                        self.all_tied_weights_keys = {k: None for k in tied}
                    except Exception:  # pragma: no cover
                        self.all_tied_weights_keys = {}
                return _orig_mark(self, *a, **kw)
            _tmu.PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied

            # Also patch the _finalize_model_loading call that invokes
            # ``model.tie_weights(missing_keys=..., recompute_mapping=...)``.
            # Legacy remote code does not accept those kwargs.
            _orig_final = _tmu.PreTrainedModel._finalize_model_loading
            def _safe_finalize(cls_or_self, *a, **kw):
                try:
                    return _orig_final(cls_or_self, *a, **kw)
                except TypeError as e:
                    if "tie_weights" in str(e) or "missing_keys" in str(e):
                        # Fall back: try without kwargs
                        model = a[0] if a else kw.get("model")
                        if model is not None:
                            try:
                                model.tie_weights()
                            except Exception:
                                pass
                        return None
                    raise
            _tmu.PreTrainedModel._finalize_model_loading = _safe_finalize
            _tmu.PreTrainedModel._molmo_patched = True

        from transformers import AutoModelForCausalLM
        import torch as _torch

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if not hasattr(model, "all_tied_weights_keys"):
            tied = getattr(model, "_tied_weights_keys", None) or []
            model.all_tied_weights_keys = {k: None for k in tied}
        try:
            model = model.to("cuda" if _torch.cuda.is_available() else "cpu")
        except Exception:
            pass
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )

    elif family == "fastvlm":
        # Apple FastVLM: trust_remote_code; architecture is LlavaQwen2
        # with a FastViT-based vision encoder.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        try:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True,
            )
        except Exception:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )

    elif family == "bunny":
        from transformers import AutoModelForCausalLM, AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
            cfg.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=cfg, torch_dtype=torch_dtype,
            low_cpu_mem_usage=False, trust_remote_code=True,
        ).eval().cuda()
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )

    elif family == "ovis":
        # Ovis2 is natively supported in transformers>=5.5 as
        # Ovis2ForConditionalGeneration. Bypass the custom remote-code
        # (which tries to re-register "aimv2_visual_tokenizer" and collides
        # with the built-in config) and use the native class directly.
        from transformers import Ovis2ForConditionalGeneration

        model = Ovis2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "idefics3":
        from transformers import Idefics3ForConditionalGeneration

        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "idefics2":
        from transformers import Idefics2ForConditionalGeneration

        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "pixtral":
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "phi4mm":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )

    elif family in ("minicpm_l3", "minicpm_v45"):
        from transformers import AutoModel
        import transformers.integrations.accelerate as _accel_mod
        _orig_init_iad = _accel_mod._init_infer_auto_device_map

        def _safe_init_iad(model, *a, **kw):
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}
            return _orig_init_iad(model, *a, **kw)

        _accel_mod._init_infer_auto_device_map = _safe_init_iad

        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )

        _accel_mod._init_infer_auto_device_map = _orig_init_iad

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        _special_map = {
            "im_start_id": "<image>", "im_end_id": "</image>",
            "slice_start_id": "<slice>", "slice_end_id": "</slice>",
        }
        tok = processor.tokenizer
        for attr, token_str in _special_map.items():
            if not hasattr(tok, attr):
                tid = tok.convert_tokens_to_ids(token_str)
                if tid is not None:
                    setattr(tok, attr, tid)
        if not hasattr(tok, "bos_id"):
            tok.bos_id = tok.bos_token_id
        if not hasattr(tok, "eos_id"):
            tok.eos_id = tok.eos_token_id
        if not hasattr(tok, "eot_id"):
            eot = tok.convert_tokens_to_ids("<|eot_id|>")
            if eot is not None and eot != tok.unk_token_id:
                tok.eot_id = eot
        model.processor = processor

    elif family == "llava_onevision":
        from transformers import LlavaOnevisionForConditionalGeneration
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "video_llava":
        from transformers import VideoLlavaForConditionalGeneration
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "vip_llava":
        from transformers import VipLlavaForConditionalGeneration
        model = VipLlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "llava15":
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "glm4v":
        # GLM-4V-9B (THUDM): EVA2-CLIP + GLU + ChatGLM-4. Uses chat template via
        # tokenizer.apply_chat_template; tokenizer doubles as image processor
        # (it accepts the PIL image directly through the chat template "image" key).
        # Requires transformers~=4.45 (compat_env) due to a `dtype=str` bug in the
        # custom modeling code at higher versions.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Force torch.float16 (not "float16" string) — GLM-4V's modeling_chatglm
        # passes the dtype straight to nn.Embedding(...) which only accepts a
        # torch.dtype, not a string.
        _dtype = torch_dtype
        if isinstance(_dtype, str):
            _dtype = getattr(torch, _dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        # Return the tokenizer as the "processor" — wrapper code calls
        # tokenizer.apply_chat_template([{role,image,content}], ...).
        processor = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )

    elif family == "glm41v":
        from transformers import Glm4vForConditionalGeneration

        model = Glm4vForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "varco_vision":
        from transformers import LlavaOnevisionForConditionalGeneration

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "aya_vision":
        from transformers import AyaVisionForConditionalGeneration

        model = AyaVisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "molmo2":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )

    elif family == "llava_next_video":
        from transformers import LlavaNextVideoForConditionalGeneration
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "llava_next":
        from transformers import LlavaNextForConditionalGeneration

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif family == "internvl":
        from transformers import AutoModel, AutoTokenizer
        from transformers.modeling_utils import PreTrainedModel

        _orig_finalize = getattr(PreTrainedModel, "_finalize_model_loading", None)

        @staticmethod
        def _patched_finalize(model, *args, **kwargs):
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}
            return _orig_finalize(model, *args, **kwargs)

        if _orig_finalize is not None:
            PreTrainedModel._finalize_model_loading = _patched_finalize

        # Monkey-patch torch.linspace to avoid meta-tensor .item() crash
        # in InternVL3.5's custom modeling_intern_vit.py
        _orig_linspace = torch.linspace
        def _safe_linspace(*args, **kwargs):
            result = _orig_linspace(*args, **kwargs)
            if result.device.type == "meta":
                import math
                start, end = float(args[0]), float(args[1])
                steps = int(args[2]) if len(args) > 2 else kwargs.get("steps", 2)
                return torch.tensor(
                    [start + (end - start) * i / max(steps - 1, 1) for i in range(steps)]
                )
            return result
        torch.linspace = _safe_linspace

        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            use_flash_attn=True, trust_remote_code=True,
        ).eval().cuda()

        torch.linspace = _orig_linspace

        if _orig_finalize is not None:
            PreTrainedModel._finalize_model_loading = _orig_finalize
        processor = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False,
        )
        if processor.pad_token_id is None:
            processor.pad_token_id = processor.eos_token_id
        eos = processor.eos_token_id
        for obj in (model, getattr(model, "language_model", None)):
            gc = getattr(obj, "generation_config", None)
            if gc is not None and gc.pad_token_id is None:
                gc.pad_token_id = eos

    elif family == "nemotron_vl":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.modeling_utils import PreTrainedModel

        _orig_init = PreTrainedModel.__init_subclass__
        _orig_finalize_nem = getattr(PreTrainedModel, "_finalize_model_loading", None)

        @staticmethod
        def _patched_finalize_nem(mdl, *args, **kwargs):
            if not hasattr(mdl, "all_tied_weights_keys"):
                mdl.all_tied_weights_keys = {}
            return _orig_finalize_nem(mdl, *args, **kwargs)

        if _orig_finalize_nem is not None:
            PreTrainedModel._finalize_model_loading = _patched_finalize_nem

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype,
            trust_remote_code=True, attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).eval().cuda()

        if _orig_finalize_nem is not None:
            PreTrainedModel._finalize_model_loading = _orig_finalize_nem

        radio = getattr(model, "vision_model", None)
        radio = getattr(radio, "radio_model", radio) if radio is not None else None
        if radio is not None:
            radio.register_buffer(
                "summary_idxs",
                torch.tensor([0, 1, 2], dtype=torch.long, device="cuda"),
            )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor._nemotron_tokenizer = tokenizer

    else:
        raise ValueError(f"Unsupported model family: {family}")

    logger.info("Loaded %s model from %s (family=%s)", type(model).__name__, model_path, family)
    return model, processor, family


# ---------------------------------------------------------------------------
# Input preparation -- discovery (PIL image + text prompt)
# ---------------------------------------------------------------------------

def prepare_discovery_inputs(
    processor,
    image: Image.Image,
    prompt: str,
    model_family: str,
    device,
) -> dict:
    """Prepare model inputs from a PIL image and text prompt.

    Supported families: ``qwen2vl``, ``phi35v``.
    For ``internvl`` use :func:`run_discovery_forward` instead.
    """
    if model_family == "gemma4":
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,
        )

    elif model_family in ("qwen3vl", "qwen35"):
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,
        )

    elif model_family == "qwen2vl":
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )

    elif model_family == "phi35v":
        msgs = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
        text = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text, [image], return_tensors="pt")

    elif model_family == "llama32v":
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")

    elif model_family in ("idefics3", "idefics2", "llava_next", "llava_next_video", "llava_onevision", "vip_llava", "pixtral", "varco_vision", "aya_vision", "molmo2"):
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")

    elif model_family == "fastvlm":
        # Apple FastVLM (LlavaQwen2) uses an ``<image>`` token in the prompt;
        # its custom tokenizer + processor don't expose apply_chat_template.
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        from transformers import AutoTokenizer as _AT
        tok = processor if hasattr(processor, "encode") else getattr(processor, "tokenizer", None)
        if tok is None:
            tok = _AT.from_pretrained(
                "apple/FastVLM-7B", trust_remote_code=True,
            )
        text = (
            "<|im_start|>user\n<image>\n" + prompt +
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        # FastVLM expects tokenised prompt + pixel_values (image tensor).
        input_ids = tok(text, return_tensors="pt").input_ids
        from torchvision import transforms as _T
        pix = _T.Compose([
            _T.Resize(256), _T.CenterCrop(256), _T.ToTensor(),
        ])(image).unsqueeze(0)
        inputs = {"input_ids": input_ids, "pixel_values": pix}

    elif model_family == "gemma3":
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )

    elif model_family == "kimi_vl":
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        # Kimi-VL follows the standard chat-template path.
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")

    elif model_family == "ovis":
        # Ovis2 uses its own chat interface; fall back to the standard
        # chat-template path which most remote-code processors support.
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        try:
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(text=text, images=[image], return_tensors="pt")
        except Exception:
            # Last-ditch fallback: raw text + image list
            inputs = processor(text=prompt, images=[image], return_tensors="pt")

    elif model_family in ("minicpm_l3", "minicpm_v45"):
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        msgs = [{"role": "user", "content": "(<image>./</image>)\n" + prompt}]
        text = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text, [image], return_tensors="pt", max_length=2048)

    elif model_family in ("bunny", "video_llava"):
        text = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = processor(text=text, images=image, return_tensors="pt")

    elif model_family == "llava15":
        # LLaVA-1.5: standard HF LlavaForConditionalGeneration uses
        # processor.apply_chat_template + processor(text, images, ...) flow.
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")

    elif model_family == "glm41v":
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
        msgs = [{"role": "user", "content": user_content}]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )

    elif model_family == "glm4v":
        # GLM-4V-9B uses its tokenizer's apply_chat_template with role=user,
        # image=PIL image, content=prompt → returns input_ids/attention_mask/
        # position_ids/images all packaged. The chat template injects the
        # special <|begin_of_image|>...<|end_of_image|> tokens that the model
        # uses to splice in vision features.
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        tok = processor  # GLM-4V exposes the tokenizer as the "processor".
        msgs = [{"role": "user", "image": image, "content": prompt}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        )

    elif model_family == "deepseek_vl2":
        # DeepSeek-VL2's custom processor expects a
        # ``conversations``/``images``/``force_batchify`` API — adapt here.
        from PIL import Image as _PILImage  # re-import for type safety
        image = _cap_image_size(image, _QWEN3_MAX_DIM)
        conversations = [
            {"role": "<|User|>", "content": f"<image>\n{prompt}", "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        if hasattr(processor, "__call__"):
            try:
                inputs = processor(
                    conversations=conversations, images=[image],
                    force_batchify=True, system_prompt="",
                )
                # Most VLChatProcessor outputs are dict-like already.
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                return inputs
            except TypeError:
                pass
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")

    else:
        raise ValueError(
            f"prepare_discovery_inputs does not support '{model_family}'. "
            "Use run_discovery_forward() instead."
        )

    # `inputs` may be a HuggingFace BatchEncoding (has .to) OR a plain dict
    # (e.g. MiniCPM, GLM-4V manual builds). Handle both.
    if hasattr(inputs, "to"):
        return inputs.to(device)
    def _to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, list):
            return [_to_device(item) for item in obj]
        return obj

    moved: dict = {}
    for k, v in inputs.items():
        moved[k] = _to_device(v)
    return moved


def run_discovery_forward(model, processor, model_family, image, prompt, device):
    """Run a single forward pass for activation collection.

    For ``qwen2vl`` and ``phi35v`` this prepares inputs and calls
    ``model(**inputs)``.  For ``internvl`` it calls ``model.chat()``
    with ``max_new_tokens=1`` (the hooks still capture activations).

    Caller should wrap in ``torch.no_grad()``.
    """
    if model_family == "internvl":
        pixel_values = load_image_internvl(image).to(torch.bfloat16).to(device)
        question = "<image>\n" + prompt
        gen_cfg = dict(max_new_tokens=1, do_sample=False)
        model.chat(processor, pixel_values, question, gen_cfg)
    elif model_family in ("minicpm_l3", "minicpm_v45"):
        inputs = prepare_discovery_inputs(
            processor, image, prompt, model_family, device,
        )
        if "position_ids" not in inputs:
            seq_len = inputs["input_ids"].shape[-1]
            inputs["position_ids"] = torch.arange(
                seq_len, device=device,
            ).unsqueeze(0)
        model(inputs)
    else:
        inputs = prepare_discovery_inputs(
            processor, image, prompt, model_family, device,
        )
        model(**inputs)


# ---------------------------------------------------------------------------
# Image resizing for Qwen3-VL / Qwen3.5 (unbounded default resolution)
# ---------------------------------------------------------------------------

_QWEN3_MAX_DIM = 512


def _cap_image_size(img: Image.Image, max_dim: int = _QWEN3_MAX_DIM) -> Image.Image:
    """Down-scale *img* so neither dimension exceeds *max_dim*."""
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _load_and_cap(path: str, max_dim: int = _QWEN3_MAX_DIM) -> Image.Image:
    """Open an image file and cap its resolution for Qwen3 models."""
    return _cap_image_size(Image.open(path).convert("RGB"), max_dim)


# ---------------------------------------------------------------------------
# Input preparation -- generation (VLMEvalKit message format, qwen2vl only)
# ---------------------------------------------------------------------------

def _vlmeval_to_qwen_content(
    vlmeval_message: list[dict], *, use_file_uri: bool = True,
    resize_max_dim: int | None = None,
) -> list[dict]:
    """Convert VLMEvalKit ``[{type, value}, ...]`` to Qwen content dicts.

    When *use_file_uri* is True (default, for Qwen2.5-VL + ``qwen_vl_utils``),
    local paths are converted to ``file://`` URIs.  Set to False for Qwen3-VL /
    Qwen3.5 where the processor expects raw paths or PIL images.

    When *resize_max_dim* is set, local images are loaded and resized so that
    neither dimension exceeds the given value (PIL objects are returned instead
    of paths).
    """
    content: list[dict] = []
    for s in vlmeval_message:
        if s["type"] == "image":
            path = s["value"]
            if resize_max_dim and not path.startswith(("http://", "https://", "data:")):
                content.append({
                    "type": "image",
                    "image": _load_and_cap(path, resize_max_dim),
                })
            elif use_file_uri:
                if not path.startswith(("http://", "https://", "file://", "data:")):
                    if os.path.exists(path):
                        path = "file://" + os.path.abspath(path)
                content.append({"type": "image", "image": path})
            else:
                if not path.startswith(("http://", "https://", "data:")):
                    path = os.path.abspath(path)
                content.append({"type": "image", "image": path})
        elif s["type"] == "text":
            content.append({"type": "text", "text": s["value"]})
    return content


def prepare_generate_inputs(
    processor,
    vlmeval_message: list[dict],
    model_family: str,
    device,
    system_prompt: str | None = None,
):
    """Prepare model inputs from VLMEvalKit-format messages for generation.

    Currently only supports ``qwen2vl``.  Phi-3.5-vision and InternVL
    generation is handled directly in the wrapper's family-specific methods.
    """
    if model_family == "glm41v":
        from PIL import Image as _PIL_Image

        user_content = []
        for m in vlmeval_message:
            if m.get("type") == "image":
                user_content.append({"type": "image", "image": _PIL_Image.open(m["value"]).convert("RGB")})
            elif m.get("type") == "text":
                user_content.append({"type": "text", "text": m["value"]})

        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": "Answer directly without thinking."}]},
            {"role": "user", "content": user_content},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        return inputs.to(device)

    if model_family in ("gemma4", "gemma3"):
        user_content = _vlmeval_to_qwen_content(
            vlmeval_message, use_file_uri=False, resize_max_dim=_QWEN3_MAX_DIM,
        )
        messages: list[dict] = [
            {"role": "user", "content": user_content},
        ]
        extra_kwargs: dict = {}
        if model_family == "gemma4":
            extra_kwargs["enable_thinking"] = False
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            **extra_kwargs,
        )
        return inputs.to(device)

    if model_family in ("qwen3vl", "qwen35"):
        user_content = _vlmeval_to_qwen_content(
            vlmeval_message, use_file_uri=False, resize_max_dim=_QWEN3_MAX_DIM,
        )
        sys_text = system_prompt or "You are a helpful assistant."
        messages_q3: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user", "content": user_content},
        ]

        inputs = processor.apply_chat_template(
            messages_q3, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,
        )
        return inputs.to(device)

    if model_family == "qwen2vl":
        from qwen_vl_utils import process_vision_info

        user_content = _vlmeval_to_qwen_content(vlmeval_message)
        messages_q: list[dict] = []
        if system_prompt:
            messages_q.append({"role": "system", "content": system_prompt})
        messages_q.append({"role": "user", "content": user_content})

        text = processor.apply_chat_template(
            messages_q, tokenize=False, add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages_q)
        inputs = processor(
            text=[text], images=images, videos=videos,
            padding=True, return_tensors="pt",
        )
        return inputs.to(device)

    if model_family == "nemotron_vl":
        from PIL import Image as _PIL_Image

        images = []
        text_parts = []
        for m in vlmeval_message:
            if m.get("type") == "image":
                images.append(_PIL_Image.open(m["value"]).convert("RGB"))
                text_parts.append({"type": "image", "image": ""})
            elif m.get("type") == "text":
                text_parts.append({"type": "text", "text": m["value"]})

        messages_nem: list[dict] = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": text_parts},
        ]

        tokenizer = getattr(processor, "_nemotron_tokenizer", None)
        if tokenizer is None:
            raise ValueError("Nemotron processor missing _nemotron_tokenizer")
        prompt = tokenizer.apply_chat_template(
            messages_nem, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt], images=images if images else None,
            return_tensors="pt",
        )
        return inputs.to(device)

    raise ValueError(
        f"prepare_generate_inputs does not support '{model_family}'. "
        "Use model-family-specific generation in the wrapper."
    )
