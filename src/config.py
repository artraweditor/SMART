import dataclasses
from typing import Literal
from pathlib import Path
import os
import json
import sys
from platformdirs import user_config_dir, user_data_dir


version = '0.3'


def is_frozen():
    """True when running as a stand-alone (PyInstaller) build."""
    return getattr(sys, 'frozen', False)


def resource_dir():
    """Directory containing the read-only data/ and icons/ directories."""
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


# SAM2.1 networks for AI masking, converted to ONNX with
# tools/export_sam2_onnx.py: quality -> ((encoder file name, url, size,
# sha256), (decoder file name, url, size, sha256)).
# The files are in https://huggingface.co/agriggio/smart-onnx-models (pinned to
# a commit).
_HF_URL = 'https://huggingface.co/agriggio/smart-onnx-models/resolve/0a80c612c962a226e261c66c78e00354543cb3fa/'
_SAM2_URL = _HF_URL
def _sam2(q, esize=None, esha=None, dsize=None, dsha=None):
    def f(kind, size, sha):
        name = f'sam2.1_hiera_{q}_{kind}.onnx'
        return (name, _SAM2_URL + name if _SAM2_URL else None, size, sha)
    return (f('encoder', esize, esha), f('decoder', dsize, dsha))
MASK_MODELS = {
    'tiny': _sam2(
        'tiny', 109_471_931,
        '769f6278eda2fdcb53c3e247e42259ce3f4002ef8e885d838ca4f2d6345c5d0c',
        16_508_159,
        'cccf25b0b49819fb25c389841046dfbf5ac169be58d82e5cf6d6894675a43b36'),
    'small': _sam2(
        'small', 138_018_779,
        'd1eee90b60dd814dd16db38f4adcb7b7bf39686235b9afe5f00404bdde9188e5',
        16_508_159,
        'a5e2fdb10ed20763df5e2d5937bdbcb6ee108e63ea8bdb87582489360e847c76'),
    'base_plus': _sam2(
        'base_plus', 277_472_669,
        'b301c76a5f18ea409a93b0ce0de3954ecbe910fa6b328a6f25450d4c41dc6318',
        16_508_159,
        '89a0854ccf41ea96b75a1ae60cc8582936bf52dc602d4df8072f750413945c01'),
    'large': _sam2(
        'large', 852_451_097,
        'c4de4a5e5c0dfbf3434a53a23b6cb7edc2172e472becb0754f1d2840cea35960',
        16_508_159,
        '76ba67bd3f608ba9c5ef1ab3fa2f312f7754afbe96fcd1cc36d691072d56e555'),
}

# The LaMa model for AI healing: (file name, url, size, sha256).  It is
# greycard's big-lama export (Apache-2.0,
# https://huggingface.co/jessolmstead/greycard-lama) with the few ops that the
# CoreML provider rejects rewritten by tools/make_lama_coreml.py; the results
# are the same and it runs on every provider.
HEAL_MODEL = (
    'big-lama-dynamic-coreml.onnx',
    _HF_URL + 'big-lama-dynamic-coreml.onnx',
    208_852_027,
    'f7707ac0db751c9cf35ec3eee149032068862c96e3f7ffa6109eef97ec36a414',
)

# model registry for raw denoising (greycard's models, GPL-3.0, copied from
# https://huggingface.co/jessolmstead/greycard-denoise):
# tier -> (file name, url, size, sha256)
DENOISE_MODELS = {
    'small': ('denoise-v19.onnx', _HF_URL + 'denoise-v19.onnx',
              4_932_834,
              '319d024a39d3e31c95e99c91bb5c089691707cbf13f4f077c1c6e49d10235c49'),
    'medium': ('denoise-v15.onnx', _HF_URL + 'denoise-v15.onnx',
               9_037_296,
               'f5845119e0b0ee63c57aad38129aed1c8925e60ec9b7838c6b7a0fa23a84a546'),
    'large': ('denoise-v20.onnx', _HF_URL + 'denoise-v20.onnx',
              20_297_204,
              'c20aed71a45b2a984570e09d8456c54ce8eff43a334b54c7ebb480c21518f9be'),
}

# TreeNet denoisers of RawForge (MIT) for demosaiced linear DNGs, as
# repackaged in https://huggingface.co/AyedaOk/rawlab-models (copied to
# _HF_URL); same tiers as DENOISE_MODELS: tier -> (file name, url, size, sha256)
LINEAR_DENOISE_MODELS = {
    'small': (
        'treenet-light.onnx',
        _HF_URL + 'treenet-light.onnx', 1_510_483,
        '082612f3de352af7353cb3de893f8dd876b8cb942c53cf5c5f8ab4ae2cc9bf78'),
    'medium': (
        'treenet.onnx',
        _HF_URL + 'treenet.onnx', 10_073_980,
        '76a07047a33dba3fa27330cec483b43bb57ecb1c1f10db8d979355ce110e34c9'),
    'large': (
        'treenet-heavy.onnx',
        _HF_URL + 'treenet-heavy.onnx', 21_863_084,
        '0074e2f7bec359c613305811c2c4e7123f738010fb3a98ce67044c48ecfc0129'),
}


def denoise_models(kind):
    """The model registry for a kind of DNG: 'mosaic' or 'linear'."""
    return LINEAR_DENOISE_MODELS if kind == 'linear' else DENOISE_MODELS


@dataclasses.dataclass
class Config:
    # AI masking: SAM2.1 quality (tiny, small, base_plus or large) and the
    # onnxruntime provider for the image encoder (cpu, coreml, cuda, ...)
    model: str = "base_plus"
    mask_provider: str = "cpu"
    mask_color: tuple[int, int, int] = (70, 230, 50)
    background_color: tuple[int, int, int] = (127, 127, 127)
    exiftool: str = "exiftool"
    window_size: int = 1200, 800
    last_dir: str = ""
    display_icc_profile: str|None = None
    # AI healing
    heal_providers: list[str]|None = None  # onnxruntime providers
    heal_max_tile: int = 1024
    heal_seam_match: bool = True
    heal_even: bool = True
    heal_regrain: bool = True
    # raw denoising
    denoise_model: str = "medium"       # small, medium or large
    denoise_provider: str = "auto"      # onnxruntime provider
    denoise_tile: int = 1024
    denoise_margin: int = 96
    denoise_dng_bits: int = 16          # 16 (integer) or 32 (float) DNG output
    brush_radius: float = 30.0
    brush_feather: float = 0.3
    brush_opacity: float = 1.0
    brush_color: tuple[int, int, int] = (255, 60, 60)

    @staticmethod
    def models_dir():
        if is_frozen():
            # the application bundle is read-only: keep models in user space
            return Path(user_data_dir('SMART', 'ART')) / 'models'
        return Path(__file__).parent / '../models'

    def mask_quality(self):
        """The SAM2.1 quality (a key of MASK_MODELS) selected by `model`;
        also understands the names of the old PyTorch checkpoints."""
        m = self.model
        if m not in MASK_MODELS:
            m = m.removeprefix('sam2.1_hiera_').removesuffix('.pt')
        if m not in MASK_MODELS:
            raise ValueError(f'unknown masking model: {self.model} (use one '
                             f'of {", ".join(MASK_MODELS)})')
        return m

    def get_mask_model_files(self):
        """(encoder, decoder) ONNX files of the masking model."""
        enc, dec = MASK_MODELS[self.mask_quality()]
        return Config.models_dir() / enc[0], Config.models_dir() / dec[0]

    def get_heal_model_file(self):
        return Config.models_dir() / HEAL_MODEL[0]

    def get_denoise_model_file(self, tier, kind='mosaic'):
        return Config.models_dir() / denoise_models(kind)[tier][0]

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
                known = {f.name for f in dataclasses.fields(Config)}
                res = dataclasses.replace(
                    res, **{k: v for k, v in json.load(f).items()
                            if k in known})
        return res

    def save(self, filename=None):
        if filename is None:
            filename = Config.get_config_file()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as out:
            json.dump(dataclasses.asdict(self), out, indent=2, sort_keys=True)
            out.write('\n')

# end of class Config
