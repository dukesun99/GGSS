"""
Activation extraction hooks and steering hook for VLM models.

Supported architectures:
  - Qwen2.5-VL:      model.model.visual (.blocks, .merger.mlp) / model.model (.layers)
  - Qwen3-VL:        model.model.visual (.blocks, .merger.linear_fc2) / model.model.language_model (.layers)
  - Qwen3.5:         model.model.visual (.blocks, .merger.linear_fc2) / model.model.language_model (.layers)
  - Phi-3.5-vision:   model.model.vision_embed_tokens (.img_projection) / model.model.layers
  - InternVL3:        model.vision_model / model.mlp1 / model.language_model
  - Llama 3.2 Vision: model.vision_model / model.multi_modal_projector / model.language_model

Unified layer names (resolved automatically per architecture):
  - projection-mlp2     last Linear in the vision→LM projection MLP
  - projection-merger    full projection module
  - lm-layer{N}         language model layer N
  - vision-block{N}     vision encoder block N
"""

import torch


class ActivationCache:
    """Captures and stores activations from specific model layers."""

    def __init__(self):
        self.activations: dict[str, torch.Tensor] = {}
        self.hooks: list = []

    def get_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def register_hook(self, module, name: str):
        hook = module.register_forward_hook(self.get_hook(name))
        self.hooks.append(hook)
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get(self, name: str, pool: str = "mean"):
        """Get cached activation with optional pooling.

        Args:
            name: Layer name used during registration.
            pool: "mean", "first", "last", or "none".

        Returns:
            Pooled tensor or raw activation.
        """
        act = self.activations.get(name)
        if act is None:
            return None

        if len(act.shape) == 2:
            if pool == "mean":
                return act.mean(dim=0)
            elif pool == "first":
                return act[0, :]
            elif pool == "last":
                return act[-1, :]
            return act
        elif len(act.shape) == 3:
            if pool == "mean":
                return act.mean(dim=1)
            elif pool == "first":
                return act[:, 0, :]
            elif pool == "last":
                return act[:, -1, :]
            return act
        return act


def _last_linear(module: torch.nn.Module) -> torch.nn.Module:
    """Return the last ``nn.Linear`` child inside a Sequential (or last child)."""
    children = list(module.children())
    for c in reversed(children):
        if isinstance(c, torch.nn.Linear):
            return c
    return children[-1] if children else module


def get_target_module(model, layer_name: str):
    """Resolve a human-readable layer name to the corresponding nn.Module.

    Supported names:
      - projection-mlp / projection-mlp2  (projection output layer)
      - projection-merger                 (full projection module)
      - lm-layer{N}                       (language model layer N)
      - vision-block{N}                   (vision encoder block N)

    Automatically detects the model architecture.
    """

    # --- Gemma 4 layout (model.model has embed_vision, language_model, vision_tower) ---
    _inner = getattr(model, "model", model)
    if hasattr(_inner, "embed_vision") and hasattr(_inner, "language_model"):
        if layer_name in ("projection-mlp", "projection-mlp2"):
            return _inner.embed_vision.embedding_projection
        elif layer_name == "projection-merger":
            return _inner.embed_vision
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            return _inner.language_model.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            return _inner.vision_tower.encoder.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- Gemma 3 layout: model.{vision_tower, multi_modal_projector, language_model} ---
    # (Gemma3ForConditionalGeneration wraps Gemma3Model which exposes these.)
    if (
        hasattr(_inner, "multi_modal_projector")
        and hasattr(_inner, "vision_tower")
        and hasattr(_inner, "language_model")
        and not hasattr(_inner, "connector")  # disambiguate from Idefics3
    ):
        if layer_name in ("projection-mlp", "projection-mlp2"):
            proj = _inner.multi_modal_projector
            # Gemma3MultiModalProjector has mm_input_projection_weight (a
            # raw nn.Parameter) plus a Linear path depending on version.
            # Pick the last Linear child if any, else return the module.
            last = _last_linear(proj)
            return last if isinstance(last, torch.nn.Linear) else proj
        elif layer_name == "projection-merger":
            return _inner.multi_modal_projector
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            return _inner.language_model.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            vt = _inner.vision_tower
            vm = getattr(vt, "vision_model", vt)
            enc = getattr(vm, "encoder", vm)
            return enc.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- Kimi-VL layout: model.{vision_tower, multi_modal_projector, language_model}
    # (MoE DeepseekV3 text backbone; the projector is typically an MLP.)
    if (
        hasattr(model, "vision_tower")
        and hasattr(model, "multi_modal_projector")
        and hasattr(model, "language_model")
    ):
        if layer_name in ("projection-mlp", "projection-mlp2"):
            proj = model.multi_modal_projector
            last = _last_linear(proj)
            return last if isinstance(last, torch.nn.Linear) else proj
        elif layer_name == "projection-merger":
            return model.multi_modal_projector
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            lm = model.language_model
            lm_inner = getattr(lm, "model", lm)
            return lm_inner.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            vt = model.vision_tower
            vm = getattr(vt, "vision_model", vt)
            enc = getattr(vm, "encoder", vm)
            return enc.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- GLM-4V-9B: top-level has `transformer`, with
    # transformer.{embedding, encoder (40 GLMBlock), vision (EVA2CLIPModel)}.
    # The vision projector is a GLU MLP at transformer.vision.linear_proj
    # (linear_proj, dense_h_to_4h, gate_proj, dense_4h_to_h). The closest analog
    # to projection-mlp2 is dense_4h_to_h (final Linear back to hidden). ---
    if hasattr(model, "transformer") and hasattr(model.transformer, "vision") and hasattr(model.transformer, "encoder"):
        tr = model.transformer
        vis = tr.vision
        enc = tr.encoder
        if layer_name in ("projection-mlp", "projection-mlp2"):
            # The GLU has 4 Linears; dense_4h_to_h is the final h*4->h projection.
            return vis.linear_proj.dense_4h_to_h
        elif layer_name == "projection-merger":
            return vis.linear_proj
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            return enc.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            vit = vis.transformer
            return vit.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- GLM-4.1V / Qwen3-VL: model.model has .visual + .language_model.
    # visual.merger is a PatchMerger with MLP (down_proj for GLM-4.1V, linear_fc2 for Qwen3-VL).
    if hasattr(_inner, "visual") and hasattr(_inner, "language_model") and hasattr(getattr(_inner, "visual", None), "merger"):
        vis = _inner.visual
        if layer_name in ("projection-mlp", "projection-mlp2"):
            merger = vis.merger
            if hasattr(merger, "down_proj"):
                return merger.down_proj
            if hasattr(merger, "linear_fc2"):
                return merger.linear_fc2
            raise ValueError(f"Cannot find final linear in merger: {list(dict(merger.named_children()).keys())}")
        elif layer_name == "projection-merger":
            return vis.merger
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            lm = _inner.language_model
            return lm.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            return vis.blocks[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- MiniCPM-Llama3-V-2.5: top-level has vpm + resampler + llm. ---
    # Resampler is a perceiver: kv_proj (Linear 1152->4096) + MultiheadAttention.
    # The kv_proj is the closest analog to projection-mlp2 — it's the last
    # Linear layer that lifts vision features into the LM hidden space.
    if (
        hasattr(model, "vpm")
        and hasattr(model, "resampler")
        and hasattr(model, "llm")
    ):
        rs = model.resampler
        if layer_name in ("projection-mlp", "projection-mlp2"):
            return rs.kv_proj
        elif layer_name == "projection-merger":
            return rs
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            lm = model.llm
            lm_inner = getattr(lm, "model", lm)
            return lm_inner.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            vpm = model.vpm
            enc = getattr(vpm, "encoder", vpm)
            return enc.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- Ovis2 layout: top-level has visual_tokenizer + llm; the vision
    # projector ("head") sits inside visual_tokenizer. ---
    if hasattr(model, "visual_tokenizer") and hasattr(model, "llm"):
        vt = model.visual_tokenizer
        llm = model.llm
        if layer_name in ("projection-mlp", "projection-mlp2"):
            # Ovis exposes a .head Linear that maps vision tokens into the
            # LLM's vocab embedding space.
            head = getattr(vt, "head", None) or getattr(vt, "projector", None)
            if head is None:
                raise ValueError("Ovis: no visual projection head found")
            return _last_linear(head) if list(head.children()) else head
        elif layer_name == "projection-merger":
            return vt
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            lm_inner = getattr(llm, "model", llm)
            return lm_inner.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            # Ovis uses a backbone ViT inside visual_tokenizer
            bb = getattr(vt, "backbone", vt)
            vm = getattr(bb, "vision_model", bb)
            enc = getattr(vm, "encoder", vm)
            return enc.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- DeepSeek-VL2 layout: model.{vision, projector, language}  ---
    # (DeepseekVLV2ForCausalLM: `vision` is a SigLIP-style encoder, `projector`
    # is an MLP that maps vision features into the DeepseekMoE LM.)
    if (
        hasattr(model, "projector")
        and hasattr(model, "vision")
        and hasattr(model, "language")
    ):
        if layer_name in ("projection-mlp", "projection-mlp2"):
            proj = model.projector
            last = _last_linear(proj)
            return last if isinstance(last, torch.nn.Linear) else proj
        elif layer_name == "projection-merger":
            return model.projector
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            lm = model.language
            lm_inner = getattr(lm, "model", lm)
            return lm_inner.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            vt = model.vision
            # SigLIP-style: .vision_model.encoder.layers
            vm = getattr(vt, "vision_model", vt)
            enc = getattr(vm, "encoder", vm)
            return enc.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- Llama 3.2 Vision layout ---
    # Top-level has model.model.{multi_modal_projector, language_model, vision_model}
    _inner = getattr(model, "model", model)
    if hasattr(_inner, "multi_modal_projector") and hasattr(_inner, "language_model"):
        if layer_name in ("projection-mlp", "projection-mlp2", "projection-merger"):
            # LlavaNext: multi_modal_projector is an MLP (linear_1, linear_2); grab last Linear
            # Llama 3.2 Vision: multi_modal_projector is a single Linear
            proj = _inner.multi_modal_projector
            if layer_name == "projection-merger":
                return proj
            return _last_linear(proj) if list(proj.children()) else proj
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            return _inner.language_model.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            vt = getattr(_inner, "vision_model", None) or getattr(_inner, "vision_tower", None)
            if hasattr(vt, "transformer"):
                return vt.transformer.layers[idx]
            if hasattr(vt, "encoder"):
                return vt.encoder.layers[idx]
            return vt.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- Idefics3 / SmolVLM layout: model.model.{connector.modality_projection, text_model, vision_model} ---
    if hasattr(_inner, "connector") and hasattr(_inner, "text_model"):
        if layer_name in ("projection-mlp", "projection-mlp2"):
            mp = _inner.connector.modality_projection
            return _last_linear(mp) if list(mp.children()) else mp
        elif layer_name == "projection-merger":
            return _inner.connector
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            return _inner.text_model.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            return _inner.vision_model.encoder.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- LLaVA-style legacy: model.model.mm_projector (e.g. Apple FastVLM) ---
    if hasattr(_inner, "mm_projector") and hasattr(_inner, "embed_tokens"):
        if layer_name in ("projection-mlp", "projection-mlp2"):
            return _last_linear(_inner.mm_projector)
        elif layer_name == "projection-merger":
            return _inner.mm_projector
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            return _inner.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- InternVL / Nemotron-VL layout (mlp1 projector + vision_model + language_model) ---
    if hasattr(model, "mlp1") and hasattr(model, "language_model"):
        if layer_name in ("projection-mlp", "projection-mlp2"):
            return _last_linear(model.mlp1)
        elif layer_name == "projection-merger":
            return model.mlp1
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            lm = model.language_model
            lm_inner = getattr(lm, "model", None) or getattr(lm, "backbone", lm)
            return lm_inner.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            vm = model.vision_model
            enc = getattr(vm, "encoder", vm)
            return enc.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    inner = getattr(model, "model", model)

    # --- Phi-3.5-vision layout (inner has vision_embed_tokens) ---
    if hasattr(inner, "vision_embed_tokens"):
        vet = inner.vision_embed_tokens
        if layer_name in ("projection-mlp", "projection-mlp2"):
            return _last_linear(vet.img_projection)
        elif layer_name == "projection-merger":
            return vet.img_projection
        elif layer_name.startswith("lm-layer"):
            idx = int(layer_name.replace("lm-layer", ""))
            return inner.layers[idx]
        elif layer_name.startswith("vision-block"):
            idx = int(layer_name.replace("vision-block", ""))
            return vet.img_processor.vision_model.encoder.layers[idx]
        raise ValueError(f"Unknown layer: {layer_name}")

    # --- Qwen-VL family (inner has visual with merger) ---
    if hasattr(inner, "visual"):
        visual = inner.visual
        lm = getattr(inner, "language_model", inner)
    elif hasattr(model, "visual"):
        visual = model.visual
        lm = getattr(model, "model", model)
    else:
        raise ValueError("Cannot find visual module in model")

    if layer_name in ("projection-mlp", "projection-mlp2"):
        if hasattr(visual.merger, "mlp"):
            return visual.merger.mlp[2]
        return visual.merger.linear_fc2
    elif layer_name == "projection-merger":
        return visual.merger
    elif layer_name.startswith("lm-layer"):
        layer_idx = int(layer_name.replace("lm-layer", ""))
        if hasattr(lm, "layers"):
            return lm.layers[layer_idx]
        if hasattr(lm, "language_model") and hasattr(lm.language_model, "layers"):
            return lm.language_model.layers[layer_idx]
        raise ValueError("Cannot find LM layers in model")
    elif layer_name.startswith("vision-block"):
        block_idx = int(layer_name.replace("vision-block", ""))
        return visual.blocks[block_idx]

    raise ValueError(f"Unknown layer: {layer_name}")


class SteeringHook:
    """Forward hook that applies a steerer to a module's output activations."""

    def __init__(self, steerer, alpha: float = 1.0, enabled: bool = True):
        self.steerer = steerer
        self.alpha = alpha
        self.enabled = enabled
        self.hook = None

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        if isinstance(output, tuple):
            steered = self.steerer.steer(output[0], alpha=self.alpha)
            return (steered,) + output[1:]
        return self.steerer.steer(output, alpha=self.alpha)

    def register(self, module):
        self.hook = module.register_forward_hook(self)
        return self.hook

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
