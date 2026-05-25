RACES = ["Asian", "Black", "Latino", "Middle_Eastern", "White"]
GENDERS = ["man", "woman"]

DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_LAYER = "projection-mlp2"
DEFAULT_LAYERS = ["projection-mlp2"]
DEFAULT_DISCOVERY_PROMPT = "Describe this image in detail."
DISCOVERY_OCCUPATIONS = ["cook", "doctor", "lawyer", "nurse", "teacher"]
GENDER_DISCOVERY_OCCUPATIONS = ["cook", "lawyer", "teacher"]
GENDER_EVAL_OCCUPATIONS = ["nurse", "doctor"]

MODEL_PRESETS = {
    "qwen3vl": {
        "model_path": "Qwen/Qwen3-VL-4B-Instruct",
        "layer": "projection-mlp2",
        "layers": ["projection-mlp2"],
    },
    "pixtral": {
        "model_path": "mistral-community/pixtral-12b",
        "layer": "projection-mlp2",
        "layers": ["projection-mlp2"],
    },
    "llava16_vicuna_7b": {
        "model_path": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "layer": "projection-mlp2",
        "layers": ["projection-mlp2"],
    },
    "llava16_mistral_7b": {
        "model_path": "llava-hf/llava-v1.6-mistral-7b-hf",
        "layer": "projection-mlp2",
        "layers": ["projection-mlp2"],
    },
}
