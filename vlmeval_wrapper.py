"""
VLMEvalKit-compatible wrapper for steered VLM models.

Provides the unified ``SteeredQwen2VLChat`` class used by both the CEO
salary experiment and VLMEvalKit benchmark evaluation.  Despite its name
(kept for backward compatibility), the class supports **any** VLM family
registered in :mod:`steering.model_utils` (Qwen2.5-VL, Phi-3.5-vision,
InternVL3, Llama 3.2 Vision, …).

Supports steering on one or multiple layers simultaneously.

Usage (baseline):
    model = SteeredQwen2VLChat(model_path="Qwen/Qwen2.5-VL-3B-Instruct")

Usage (single-layer steering):
    model = SteeredQwen2VLChat(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        steerer_type="geo_svd",
        steerer_path="results/Qwen2.5-VL-3B-Instruct/projection-mlp2/geo_svd.pt",
        steering_layer="projection-mlp2",
        steering_alpha=0.5,
    )

Usage (multi-layer steering):
    model = SteeredQwen2VLChat(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        steering_configs=[
            {"steerer_type": "per_token_geo",
             "steerer_path": "results/.../projection-mlp2/per_token_geo.pt",
             "layer": "projection-mlp2", "alpha": 1.0},
        ],
    )

Standalone test:
    python vlmeval_wrapper.py --mode test \\
        --steerer_type geo_svd \\
        --steerer_path results/Qwen2.5-VL-3B-Instruct/projection-mlp2/geo_svd.pt \\
        --image source/ceo/1/Black_man.jpg
"""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import torch
from PIL import Image

from steering import (
    DEFAULT_MODEL_PATH,
    DEFAULT_LAYER,
    STEERER_REGISTRY,
    STEERER_FILENAMES,
    SteeringHook,
    get_target_module,
    load_steerer,
    load_vlm,
    load_image_internvl,
    detect_model_family,
    prepare_generate_inputs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure VLMEvalKit is importable
# ---------------------------------------------------------------------------

_VLMEVAL_DIR = str(Path(__file__).resolve().parent / "VLMEvalKit")
if Path(_VLMEVAL_DIR).exists() and _VLMEVAL_DIR not in sys.path:
    sys.path.insert(0, _VLMEVAL_DIR)

try:
    from vlmeval.vlm.base import BaseModel  # type: ignore
except Exception as _vlmeval_err:  # pragma: no cover
    # Fallback stub so the wrapper still works in environments where the full
    # VLMEvalKit is not importable (e.g. compat envs pinned to older
    # transformers for legacy VLMs). Only the minimal BaseModel API used by
    # the bias-experiment scripts is re-implemented here.
    logger.warning(
        "VLMEvalKit unavailable (%s); using minimal BaseModel stub.",
        _vlmeval_err,
    )

    class BaseModel:  # type: ignore[override]
        INSTALL_REQ = False
        INTERLEAVE = True
        VIDEO_LLM = False

        def __init__(self, *args, **kwargs):  # noqa: D401
            pass

        def use_custom_prompt(self, dataset):
            return False

        def dump_image(self, line, dataset=None):
            # Minimal: return the image path(s) from the line dict
            img = line.get("image_path") or line.get("image")
            return [img] if isinstance(img, str) else (img or [])

        def generate(self, message, dataset=None):
            return self.generate_inner(message, dataset)

import re

_MCQ_SUFFIX = "\nIMPORTANT: Answer with ONLY the single letter (A, B, C, or D). Do not explain."
_MCQ_PATTERN = re.compile(
    r'\b[A-D]\.\s', re.MULTILINE,
)


def _append_mcq_suffix(message: list[dict]) -> list[dict]:
    """If the message looks like an MCQ question, append a concise-answer suffix.

    This prevents verbose reasoning that confuses VLMEvalKit's answer extraction.
    """
    for m in message:
        if m["type"] == "text" and _MCQ_PATTERN.search(m["value"]):
            out = []
            for item in message:
                if item["type"] == "text":
                    out.append({"type": "text", "value": item["value"] + _MCQ_SUFFIX})
                else:
                    out.append(item)
            return out
    return message


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

class SteeredQwen2VLChat(BaseModel):
    """VLM model with optional activation steering on one or more layers.

    Inherits from VLMEvalKit's BaseModel for benchmark compatibility.
    Also usable directly via ``model.generate(message)``.

    Despite the class name (kept for backward compatibility), this wrapper
    supports any VLM family registered in :mod:`steering.model_utils`
    (Qwen2.5-VL, Phi-3.5-vision, InternVL3, Llama 3.2 Vision, …).
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        # --- legacy single-layer API (backwards compatible) ---
        steerer_type: str | None = None,
        steerer_path: str | None = None,
        steering_alpha: float = 1.0,
        steering_layer: str = DEFAULT_LAYER,
        gate_floor: float | None = None,
        # --- multi-layer API ---
        steering_configs: list[dict] | None = None,
        # --- generation ---
        max_new_tokens: int = 2048,
        temperature: float = 0.01,
        do_sample: bool = False,
        torch_dtype: str = "bfloat16",
        system_prompt: str | None = None,
        use_custom_prompt: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.verbose = verbose
        self._use_custom_prompt = use_custom_prompt

        self._steering_configs_raw = steering_configs
        self._legacy_steerer_type = steerer_type
        self._gate_floor = gate_floor

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "auto": "auto",
        }
        _dtype = dtype_map.get(torch_dtype, torch_dtype)

        logger.info("Loading model: %s", model_path)
        self.model, self.processor, self.model_family = load_vlm(
            model_path, torch_dtype=_dtype,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Unified tokenizer access (InternVL stores tokenizer as processor)
        if self.model_family == "glm4v":
            # GLM-4V exposes the tokenizer DIRECTLY via load_vlm's
            # ``processor`` return value. ChatGLM4Tokenizer in particular has a
            # ``.tokenizer`` attribute that is a tokenizers ``Encoding`` object,
            # which would shadow the real tokenizer if we picked it via hasattr.
            self._tokenizer = self.processor
        elif hasattr(self.processor, "tokenizer"):
            self._tokenizer = self.processor.tokenizer
        else:
            self._tokenizer = self.processor

        self.generate_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if do_sample:
            self.generate_kwargs["temperature"] = temperature

        self.steering_hooks: dict[str, SteeringHook] = {}
        self._hook_handles: dict[str, object] = {}

        configs = self._resolve_configs(
            steering_configs, steerer_type, steerer_path, steering_alpha, steering_layer,
            gate_floor=gate_floor,
        )
        for cfg in configs:
            self._install_steering(
                cfg["steerer_type"],
                cfg["steerer_path"],
                cfg["alpha"],
                cfg["layer"],
                gate_floor=cfg.get("gate_floor"),
            )
        if not configs:
            logger.info("No steering configured -- running as baseline.")

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_configs(
        steering_configs: list[dict] | None,
        steerer_type: str | None,
        steerer_path: str | None,
        steering_alpha: float,
        steering_layer: str,
        gate_floor: float | None = None,
    ) -> list[dict]:
        """Normalise legacy single-layer args and multi-layer list into
        a uniform list of ``{steerer_type, steerer_path, layer, alpha}``."""
        if steering_configs is not None:
            out: list[dict] = []
            for i, cfg in enumerate(steering_configs):
                for key in ("steerer_type", "steerer_path", "layer"):
                    if key not in cfg:
                        raise ValueError(
                            f"steering_configs[{i}] is missing required key '{key}'"
                        )
                entry = {
                    "steerer_type": cfg["steerer_type"],
                    "steerer_path": cfg["steerer_path"],
                    "layer": cfg["layer"],
                    "alpha": cfg.get("alpha", 1.0),
                }
                if "gate_floor" in cfg:
                    entry["gate_floor"] = cfg["gate_floor"]
                out.append(entry)
            return out

        if steerer_type is not None and steerer_path is not None:
            entry = {
                "steerer_type": steerer_type,
                "steerer_path": steerer_path,
                "layer": steering_layer,
                "alpha": steering_alpha,
            }
            if gate_floor is not None:
                entry["gate_floor"] = gate_floor
            return [entry]

        if steerer_type is not None or steerer_path is not None:
            raise ValueError(
                "Both steerer_type and steerer_path must be provided together, "
                f"got steerer_type={steerer_type}, steerer_path={steerer_path}"
            )

        return []

    # ------------------------------------------------------------------

    def _install_steering(
        self,
        steerer_type: str,
        steerer_path: str,
        alpha: float,
        layer: str,
        gate_floor: float | None = None,
        **kwargs,
    ):
        if layer in self.steering_hooks:
            raise ValueError(
                f"A steering hook is already installed on layer '{layer}'. "
                "Each layer can only have one steerer."
            )

        logger.info("Loading steerer: %s from %s (layer=%s)", steerer_type, steerer_path, layer)
        steerer = load_steerer(steerer_type, steerer_path, device=str(self.device))

        if gate_floor is not None and hasattr(steerer, "gate_floor"):
            steerer.gate_floor = gate_floor
            logger.info("  gate_floor overridden to %.2f", gate_floor)

        if kwargs.get("kappa") is not None and hasattr(steerer, "kappa"):
            steerer.kappa = kwargs["kappa"]
            logger.info("  kappa overridden to %.2f", steerer.kappa)
        if kwargs.get("ablation_no_slerp") and hasattr(steerer, "ablation_no_slerp"):
            steerer.ablation_no_slerp = True
            logger.info("  ablation_no_slerp enabled")
        if kwargs.get("ablation_no_norm") and hasattr(steerer, "ablation_no_norm"):
            steerer.ablation_no_norm = True
            logger.info("  ablation_no_norm enabled")

        meta = getattr(steerer, "metadata", {})
        stored_model = meta.get("model_name", "")
        stored_layer = meta.get("layer_name", "")
        if stored_model and stored_model != self.model_path:
            logger.warning(
                "Steerer was fitted on model '%s' but running on '%s'",
                stored_model,
                self.model_path,
            )
        if stored_layer and stored_layer != layer:
            logger.warning(
                "Steerer was fitted on layer '%s' but hooking into '%s'",
                stored_layer,
                layer,
            )

        hook = SteeringHook(steerer, alpha=alpha)
        target_module = get_target_module(self.model, layer)
        handle = hook.register(target_module)

        self.steering_hooks[layer] = hook
        self._hook_handles[layer] = handle

        logger.info(
            "Hook installed on '%s' with alpha=%s  steerer metadata: %s",
            layer, alpha, meta,
        )

    # ------------------------------------------------------------------
    # VLMEvalKit BaseModel API
    # ------------------------------------------------------------------

    def use_custom_prompt(self, dataset):
        return self._use_custom_prompt

    def build_prompt(self, line, dataset=None):
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=question))
        return message

    def generate_inner(self, message: list[dict], dataset: str | None = None) -> str:
        """Core generation method required by VLMEvalKit's BaseModel."""
        for hook in self.steering_hooks.values():
            steerer = hook.steerer
            if hasattr(steerer, "clear_cache"):
                steerer.clear_cache()

        msg = _append_mcq_suffix(message)

        if self.verbose:
            print(f"\033[31m{msg}\033[0m")

        if self.model_family == "internvl":
            response = self._generate_internvl(msg)
        elif self.model_family == "phi35v":
            response = self._generate_phi35v(msg)
        elif self.model_family == "llama32v":
            response = self._generate_llama32v(msg)
        elif self.model_family == "glm41v":
            response = self._generate_glm41v(msg)
        elif self.model_family in ("gemma4", "gemma3"):
            response = self._generate_gemma4(msg)
        elif self.model_family == "nemotron_vl":
            response = self._generate_nemotron(msg)
        elif self.model_family in ("idefics3", "idefics2", "llava_next", "llava_next_video", "llava_onevision", "vip_llava", "pixtral", "varco_vision", "aya_vision", "molmo2"):
            response = self._generate_hf_native(msg)
        elif self.model_family in ("llava15", "bunny", "video_llava"):
            response = self._generate_llava15(msg)
        elif self.model_family in ("minicpm_l3", "minicpm_v45"):
            response = self._generate_minicpm_l3(msg)
        elif self.model_family == "glm4v":
            response = self._generate_glm4v(msg)
        else:
            response = self._generate_qwen(msg)

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response

    # ------------------------------------------------------------------
    # Family-specific generation
    # ------------------------------------------------------------------

    def _generate_qwen(self, message: list[dict]) -> str:
        inputs = prepare_generate_inputs(
            processor=self.processor,
            vlmeval_message=message,
            model_family=self.model_family,
            device=self.device,
            system_prompt=self.system_prompt,
        )
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _generate_gemma4(self, message: list[dict]) -> str:
        inputs = prepare_generate_inputs(
            processor=self.processor,
            vlmeval_message=message,
            model_family=self.model_family,
            device=self.device,
            system_prompt=self.system_prompt,
        )
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(
            generated_ids[0], skip_special_tokens=True,
        )
        return response.strip()

    def _generate_glm41v(self, message: list[dict]) -> str:
        import re
        gen_kwargs = dict(self.generate_kwargs)
        gen_kwargs["max_new_tokens"] = max(gen_kwargs.get("max_new_tokens", 512), 512)
        inputs = prepare_generate_inputs(
            processor=self.processor,
            vlmeval_message=message,
            model_family=self.model_family,
            device=self.device,
            system_prompt=self.system_prompt,
        )
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(
            generated_ids[0], skip_special_tokens=True,
        )
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if response.startswith("<think>"):
            response = ""
        for tag in ("<answer>", "</answer>", "<|begin_of_box|>", "<|end_of_box|>"):
            response = response.replace(tag, "")
        return response.strip()

    def _generate_nemotron(self, message: list[dict]) -> str:
        inputs = prepare_generate_inputs(
            processor=self.processor,
            vlmeval_message=message,
            model_family=self.model_family,
            device=self.device,
            system_prompt=self.system_prompt,
        )
        gen_kwargs = dict(self.generate_kwargs)
        gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs.get("pixel_values"),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

    def _generate_phi35v(self, message: list[dict]) -> str:
        images: list[Image.Image] = []
        content = ""
        img_idx = 0
        for m in message:
            if m["type"] == "image":
                img_idx += 1
                images.append(Image.open(m["value"]).convert("RGB"))
                content += f"<|image_{img_idx}|>\n"
            elif m["type"] == "text":
                content += m["value"]

        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": content})

        text = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        if images:
            inputs = self.processor(text, images, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                eos_token_id=self._tokenizer.eos_token_id,
                max_new_tokens=self.generate_kwargs["max_new_tokens"],
                do_sample=self.generate_kwargs.get("do_sample", False),
                use_cache=False,
            )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _generate_llama32v(self, message: list[dict]) -> str:
        images: list[Image.Image] = []
        text_parts: list[str] = []
        for m in message:
            if m["type"] == "image":
                images.append(Image.open(m["value"]).convert("RGB"))
            elif m["type"] == "text":
                text_parts.append(m["value"])

        content: list[dict] = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": "\n".join(text_parts)})

        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        if images:
            inputs = self.processor(text=text, images=images, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.generate_kwargs["max_new_tokens"],
                do_sample=self.generate_kwargs.get("do_sample", False),
            )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _generate_minicpm_l3(self, message: list[dict]) -> str:
        """Generation path for MiniCPM-Llama3-V-2.5 via its bespoke .chat() API.

        Architecture: SigLIP vpm + perceiver resampler + Llama-3-Instruct LM.
        The model exposes ``model.chat(image=PIL, msgs=[...], tokenizer=...)``
        which handles slicing, prompt templating, and decoding internally.
        ``self.processor`` is the full AutoProcessor; the tokenizer is
        ``self.processor.tokenizer`` (used for decoding inside .chat()).
        """
        from PIL import Image as _Image

        images: list = []
        text_parts: list[str] = []
        for m in message:
            if m["type"] == "image":
                img = _Image.open(m["value"]).convert("RGB")
                w, h = img.size
                if max(w, h) > 448:
                    s = 448 / max(w, h)
                    img = img.resize((int(w * s), int(h * s)), _Image.LANCZOS)
                images.append(img)
            elif m["type"] == "text":
                text_parts.append(m["value"])

        prompt_text = "\n".join(text_parts)
        msgs = [{"role": "user", "content": prompt_text}]

        with torch.no_grad():
            response = self.model.chat(
                image=images[0] if images else None,
                msgs=msgs,
                tokenizer=self._tokenizer,
                processor=self.processor,
                sampling=False,
                max_new_tokens=self.generate_kwargs.get("max_new_tokens", 256),
            )
        if isinstance(response, tuple):
            response = response[0]
        return response.strip() if isinstance(response, str) else str(response)

    def _generate_llava15(self, message: list[dict]) -> str:
        """Generation path for LLaVA-1.5-7B (Vicuna-1.5 + CLIP-ViT-L/14).

        We deliberately bypass ``processor.apply_chat_template``: the official
        Vicuna chat template prepends a long "A chat between..." preamble that
        causes the model to produce uniform/non-discriminating answers on the
        CEO-salary task. The raw template (just ``USER: <image>\\n{prompt} ASSISTANT:``)
        is what the original LLaVA paper uses for inference and what surfaces
        the model's underlying bias.
        """
        from PIL import Image as _Image

        images: list = []
        text_parts: list[str] = []
        for m in message:
            if m["type"] == "image":
                img = _Image.open(m["value"]).convert("RGB")
                # LLaVA-1.5 internally pads to 336x336; cap larger to avoid OOM.
                w, h = img.size
                if max(w, h) > 672:
                    s = 672 / max(w, h)
                    img = img.resize((int(w * s), int(h * s)), _Image.LANCZOS)
                images.append(img)
            elif m["type"] == "text":
                text_parts.append(m["value"])

        prompt_text = "\n".join(text_parts)
        # Raw single-turn format (no apply_chat_template, no system preamble).
        if images:
            text = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        else:
            text = f"USER: {prompt_text}\nASSISTANT:"

        inputs = self.processor(
            text=text, images=images if images else None, return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        # Cast tensors to model device & dtype where appropriate.
        casted = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if k == "pixel_values":
                    casted[k] = v.to(device, dtype=self.model.dtype)
                else:
                    casted[k] = v.to(device)
            else:
                casted[k] = v

        max_new = self.generate_kwargs.get("max_new_tokens", 256)
        with torch.no_grad():
            out = self.model.generate(
                **casted, max_new_tokens=max_new, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = out[0, casted["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _generate_glm4v(self, message: list[dict]) -> str:
        """Generation path for GLM-4V-9B (THUDM).

        Architecture: EVA2-CLIP vision tower + GLU vision projector + ChatGLM-4 LM.
        Uses the tokenizer's apply_chat_template to package
        ``[{role:'user', image:PIL, content:text}]`` into input_ids/attention_mask/
        position_ids/images, then calls model.generate. Note: ``self.processor``
        is the tokenizer in this family.
        """
        from PIL import Image as _Image

        images: list = []
        text_parts: list[str] = []
        for m in message:
            if m["type"] == "image":
                img = _Image.open(m["value"]).convert("RGB")
                # GLM-4V handles its own resize internally — but cap dim to keep
                # memory predictable.
                w, h = img.size
                if max(w, h) > 1120:
                    s = 1120 / max(w, h)
                    img = img.resize((int(w * s), int(h * s)), _Image.LANCZOS)
                images.append(img)
            elif m["type"] == "text":
                text_parts.append(m["value"])

        prompt_text = "\n".join(text_parts)
        # GLM-4V's chat template only takes one image (and inserts <|begin_of_image|>
        # tokens automatically). Use first image if present.
        msgs = [{"role": "user", "image": images[0] if images else None,
                 "content": prompt_text}]
        if images:
            inputs = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            )
        else:
            # Text-only fallback (no <image> placeholder needed).
            msgs_no_img = [{"role": "user", "content": prompt_text}]
            inputs = self.processor.apply_chat_template(
                msgs_no_img, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            )
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        max_new = self.generate_kwargs.get("max_new_tokens", 256)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
            )
        # Strip the prompt prefix from the output ids.
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()

    def _generate_hf_native(self, message: list[dict]) -> str:
        """Generation path for HF-native VLMs (Idefics3, LLaVA-Next, SmolVLM)."""
        from PIL import Image as _Image

        _QWEN3_MAX_DIM = 512  # cap for consistency with other families

        def _cap(img):
            w, h = img.size
            if w <= _QWEN3_MAX_DIM and h <= _QWEN3_MAX_DIM:
                return img
            scale = _QWEN3_MAX_DIM / max(w, h)
            return img.resize((int(w * scale), int(h * scale)), _Image.LANCZOS)

        images: list = []
        text_parts: list[str] = []
        for m in message:
            if m["type"] == "image":
                images.append(_cap(_Image.open(m["value"]).convert("RGB")))
            elif m["type"] == "text":
                text_parts.append(m["value"])

        content: list[dict] = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": "\n".join(text_parts)})

        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        if images:
            inputs = self.processor(
                text=text, images=images, return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.generate_kwargs["max_new_tokens"],
                do_sample=self.generate_kwargs.get("do_sample", False),
            )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _generate_internvl(self, message: list[dict]) -> str:
        image_path = None
        text_parts: list[str] = []
        for m in message:
            if m["type"] == "image":
                image_path = m["value"]
            elif m["type"] == "text":
                text_parts.append(m["value"])
        text = "\n".join(text_parts)

        pixel_values = None
        if image_path:
            pixel_values = load_image_internvl(image_path).to(
                torch.bfloat16
            ).to(self.device)

        question = ("<image>\n" + text) if pixel_values is not None else text

        gen_cfg = dict(
            max_new_tokens=self.generate_kwargs["max_new_tokens"],
            do_sample=self.generate_kwargs.get("do_sample", False),
        )
        response = self.model.chat(self.processor, pixel_values, question, gen_cfg)
        return response

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def set_steering_alpha(self, alpha: float, layer: str | None = None):
        """Set alpha for a specific layer or all layers."""
        if layer is not None:
            if layer in self.steering_hooks:
                self.steering_hooks[layer].alpha = alpha
        else:
            for hook in self.steering_hooks.values():
                hook.alpha = alpha

    def enable_steering(self, layer: str | None = None):
        """Enable steering on a specific layer or all layers."""
        if layer is not None:
            if layer in self.steering_hooks:
                self.steering_hooks[layer].enabled = True
        else:
            for hook in self.steering_hooks.values():
                hook.enabled = True

    def disable_steering(self, layer: str | None = None):
        """Disable steering on a specific layer or all layers."""
        if layer is not None:
            if layer in self.steering_hooks:
                self.steering_hooks[layer].enabled = False
        else:
            for hook in self.steering_hooks.values():
                hook.enabled = False

    def remove_steering(self, layer: str | None = None):
        """Remove steering hook(s). Pass *layer* to remove one, or ``None`` for all."""
        if layer is not None:
            handle = self._hook_handles.pop(layer, None)
            if handle is not None:
                handle.remove()
            self.steering_hooks.pop(layer, None)
        else:
            for handle in self._hook_handles.values():
                handle.remove()
            self._hook_handles.clear()
            self.steering_hooks.clear()

    @property
    def steering_hook(self) -> SteeringHook | None:
        """Backwards-compatible accessor returning the first (or only) hook."""
        if self.steering_hooks:
            return next(iter(self.steering_hooks.values()))
        return None

    def __repr__(self):
        n_hooks = len(self.steering_hooks)
        if n_hooks == 0:
            return (
                f"SteeredQwen2VLChat(model={self.model_path}, "
                f"family={self.model_family}, status=baseline)"
            )
        layers_str = ", ".join(
            f"{ln}(a={h.alpha},{'on' if h.enabled else 'off'})"
            for ln, h in self.steering_hooks.items()
        )
        return (
            f"SteeredQwen2VLChat(model={self.model_path}, "
            f"family={self.model_family}, layers=[{layers_str}])"
        )


# ---------------------------------------------------------------------------
# Helper: resolve steerer path from directory layout
# ---------------------------------------------------------------------------

def resolve_steerer_dir(
    steerer_dir: str, model_path: str, layer: str, mode: str = "race",
) -> Path:
    """Resolve ``<steerer_dir>/<model_short_name>/<layer_tag>/``.

    When *mode* is ``"gender"``, the layer component is suffixed with
    ``_gender`` so that race and gender steerers coexist without conflict.
    """
    short = model_path.rstrip("/").split("/")[-1]
    layer_tag = layer if mode == "race" else f"{layer}_{mode}"
    return Path(steerer_dir) / short / layer_tag


def build_steering_configs(
    steerer_dir: str,
    model_path: str,
    layers: list[str],
    steerer_type: str,
    alpha: float = 1.0,
    mode: str = "race",
    gate_floor: float | None = None,
) -> list[dict]:
    """Build a ``steering_configs`` list for multi-layer steering."""
    configs: list[dict] = []
    for layer in layers:
        base = resolve_steerer_dir(steerer_dir, model_path, layer, mode=mode)
        filename = STEERER_FILENAMES.get(steerer_type)
        if filename is None:
            continue
        path = base / filename
        if path.exists():
            entry = {
                "steerer_type": steerer_type,
                "steerer_path": str(path),
                "layer": layer,
                "alpha": alpha,
            }
            if gate_floor is not None:
                entry["gate_floor"] = gate_floor
            configs.append(entry)
    return configs


def get_steered_model_configs(
    steerer_dir: str = "results",
    model_path: str = DEFAULT_MODEL_PATH,
    layer: str = DEFAULT_LAYER,
    alphas: list[float] | None = None,
    mode: str = "race",
) -> dict:
    """Generate model config dict for VLMEvalKit's config.py (single-layer)."""
    if alphas is None:
        alphas = [1.0]

    from functools import partial

    base = resolve_steerer_dir(steerer_dir, model_path, layer, mode=mode)
    configs: dict = {}

    configs["Baseline"] = partial(SteeredQwen2VLChat, model_path=model_path)

    for steerer_type, filename in STEERER_FILENAMES.items():
        path = base / filename
        if not path.exists():
            continue
        for alpha in alphas:
            alpha_str = f"a{alpha}".replace(".", "p")
            name = f"{steerer_type}_{alpha_str}"
            configs[name] = partial(
                SteeredQwen2VLChat,
                model_path=model_path,
                steerer_type=steerer_type,
                steerer_path=str(path),
                steering_alpha=alpha,
                steering_layer=layer,
            )

    return configs


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Steered VLM wrapper for VLMEvalKit"
    )
    parser.add_argument("--mode", choices=["test", "list"], default="test")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--steerer_type", default=None, choices=list(STEERER_REGISTRY.keys())
    )
    parser.add_argument("--steerer_path", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "list":
        configs = get_steered_model_configs()
        print(f"\nAvailable steered model configurations ({len(configs)}):\n")
        for name in configs:
            print(f"  {name}")
        return

    if args.mode == "test":
        model = SteeredQwen2VLChat(
            model_path=args.model_path,
            steerer_type=args.steerer_type,
            steerer_path=args.steerer_path,
            steering_alpha=args.alpha,
            steering_layer=args.layer,
            verbose=args.verbose,
        )
        print(f"\n{model}\n")

        msg: list[dict] = []
        if args.image:
            msg.append({"type": "image", "value": args.image})
        msg.append({"type": "text", "value": args.prompt})

        print(f"Prompt: {args.prompt}")
        if args.image:
            print(f"Image: {args.image}")
        print("-" * 60)

        response = model.generate(msg)
        print(f"Response: {response}")
        print("-" * 60)

        if model.steering_hooks:
            model.disable_steering()
            response_baseline = model.generate(msg)
            model.enable_steering()
            print(f"Baseline (steering off): {response_baseline}")
            print("-" * 60)


if __name__ == "__main__":
    main()
