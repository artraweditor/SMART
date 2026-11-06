import dataclasses
from typing import Literal
from pathlib import Path
import os
import json
import sys
from platformdirs import user_config_dir, user_data_dir


def is_frozen():
    """True when running as a stand-alone (PyInstaller) build."""
    return getattr(sys, 'frozen', False)


def resource_dir():
    """Directory containing the read-only data/ and icons/ directories."""
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent

version = '0.2'

# SAM2.1 checkpoints for AI masking: quality -> (file name, url, size, sha256)
# (the checksum is not known: None means "not verified", it is only printed)
_SAM2_URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/'
MASK_MODELS = {
    'tiny': ('sam2.1_hiera_tiny.pt', _SAM2_URL + 'sam2.1_hiera_tiny.pt',
             156_008_466, None),
    'small': ('sam2.1_hiera_small.pt', _SAM2_URL + 'sam2.1_hiera_small.pt',
              184_416_285, None),
    'base_plus': ('sam2.1_hiera_base_plus.pt',
                  _SAM2_URL + 'sam2.1_hiera_base_plus.pt',
                  323_606_802, None),
    'large': ('sam2.1_hiera_large.pt', _SAM2_URL + 'sam2.1_hiera_large.pt',
              898_083_611, None),
}

# model registry for AI healing: backend -> (file name, url, size, checksum)
# (checksum is a sha256; None means "not verified")
HEAL_MODELS = {
    'onnx': (
        'big-lama-dynamic-fp32.onnx',
        'https://huggingface.co/jessolmstead/greycard-lama/resolve/main/'
        'big-lama-dynamic-fp32.onnx',
        209_024_270,
        '5f39f5e80b91596fd50f3227858c97912dabc0ffbd4796f21ca688529abb8771',
    ),
    'torch': (
        'big-lama.pt',
        'https://github.com/Sanster/models/releases/download/add_big_lama/'
        'big-lama.pt',
        None,
        None,  # TODO: record the checksum at first download
    ),
}

# model registry for raw denoising (greycard's models, GPL-3.0):
# tier -> (file name, url, size, sha256)
_DENOISE_URL = 'https://huggingface.co/jessolmstead/greycard-denoise/resolve/main/'
DENOISE_MODELS = {
    'fast': ('denoise-v19.onnx', _DENOISE_URL + 'denoise-v19.onnx',
             4_932_834,
             '319d024a39d3e31c95e99c91bb5c089691707cbf13f4f077c1c6e49d10235c49'),
    'balanced': ('denoise-v15.onnx', _DENOISE_URL + 'denoise-v15.onnx',
                 9_037_296,
                 'f5845119e0b0ee63c57aad38129aed1c8925e60ec9b7838c6b7a0fa23a84a546'),
    'best': ('denoise-v20.onnx', _DENOISE_URL + 'denoise-v20.onnx',
             20_297_204,
             'c20aed71a45b2a984570e09d8456c54ce8eff43a334b54c7ebb480c21518f9be'),
}


@dataclasses.dataclass
class Config:
    device: str = "cpu"
    model: str = "sam2.1_hiera_base_plus.pt"
    model_config: str|None = None
    mask_color: tuple[int, int, int] = (70, 230, 50)
    background_color: tuple[int, int, int] = (127, 127, 127)
    exiftool: str = "exiftool"
    window_size: int = 1200, 800
    last_dir: str = ""
    display_icc_profile: str|None = None
    # AI healing
    heal_backend: str = "auto"  # "auto", "onnx" or "torch"
    heal_providers: list[str]|None = None  # onnxruntime providers
    heal_max_tile: int = 1024
    heal_seam_match: bool = True
    heal_even: bool = True
    heal_regrain: bool = True
    # raw denoising
    denoise_model: str = "balanced"     # a key of DENOISE_MODELS
    denoise_backend: str = "onnxruntime"  # or "torch"
    denoise_provider: str = "auto"      # onnxruntime provider / torch device
    denoise_tile: int = 1024
    denoise_margin: int = 96
    denoise_dng_bits: int = 16          # 16 (integer) or 32 (float) DNG output
    brush_radius: float = 30.0
    brush_feather: float = 0.3
    brush_opacity: float = 1.0
    brush_color: tuple[int, int, int] = (255, 60, 60)

    def get_model_config(self):
        if self.model_config is not None:
            return str(Config.models_dir()), self.model_config
        confmap = {
            "sam2.1_hiera_tiny.pt": "sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small.pt": "sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus.pt": "sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large.pt": "sam2.1_hiera_l.yaml",
        }
        name = confmap.get(self.model)
        if name is not None:
            return str(resource_dir() / 'data'), name
        raise ValueError(f'no model configuration found')

    @staticmethod
    def models_dir():
        if is_frozen():
            # the application bundle is read-only: keep models in user space
            return Path(user_data_dir('SMART', 'ART')) / 'models'
        return Path(__file__).parent / '../models'

    def get_model_file(self):
        return Config.models_dir() / self.model

    def get_heal_model_file(self, backend):
        return Config.models_dir() / HEAL_MODELS[backend][0]

    def get_denoise_model_file(self, tier):
        return Config.models_dir() / DENOISE_MODELS[tier][0]

    @staticmethod
    def get_config_file(legacy=False):
        if legacy:
            return os.path.join(user_config_dir(), "artpixls-SMART.json")
        else:
            return os.path.join(user_config_dir(), "ART", "SMART.json")

    @staticmethod
    def load(filename=None):
        res = Config()
        if filename is None:
            filename = Config.get_config_file()
        if os.path.exists(filename):
            with open(filename) as f:
                res = dataclasses.replace(res, **json.load(f))
        return res

    def save(self, filename=None):
        if filename is None:
            filename = Config.get_config_file()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as out:
            json.dump(dataclasses.asdict(self), out, indent=2, sort_keys=True)
            out.write('\n')

# end of class Config
