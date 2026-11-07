"""SAM2.1 image predictor on onnxruntime (no PyTorch).

Runs the two networks made by tools/export_sam2_onnx.py: the image encoder
(once per image) and the prompt encoder + mask decoder (once per click).
Mimics what sam2.SAM2ImagePredictor does around them, for the way SMART uses
it (point prompts, single mask output, mask refinement through the logits of
the previous prediction).
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

import config

SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOW = 256           # side of the low-res mask logits


class ModelMissing(Exception):
    pass


def _resize_plane(a, w, h):
    """Bilinear resize of a float32 2D array (PIL's resampling filters work
    on a support scaled by the reduction factor: antialiased)."""
    im = Image.fromarray(np.ascontiguousarray(a, dtype=np.float32), mode='F')
    return np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32)


def preprocess(image):
    """float32 HxWx3 (RGB, 0..1) -> float32 1x3x1024x1024, normalized."""
    planes = [_resize_plane(image[..., c], SIZE, SIZE) for c in range(3)]
    x = (np.stack(planes, axis=-1) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def upsample_logits(logits, w, h):
    """Low-res logits (256x256) -> HxW, bilinear (align_corners=False)."""
    return _resize_plane(logits, w, h)


def _providers(provider, cache_name):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    names = {
        'cuda': 'CUDAExecutionProvider',
        'tensorrt': 'TensorrtExecutionProvider',
        'rocm': 'ROCMExecutionProvider',
        'directml': 'DmlExecutionProvider',
        'openvino': 'OpenVINOExecutionProvider',
        'coreml': 'CoreMLExecutionProvider',
        'cpu': 'CPUExecutionProvider',
    }
    if provider in (None, '', 'auto'):
        provider = 'cpu'    # the default: see config.mask_provider
    p = names.get(provider, provider)
    if p not in avail:
        raise ValueError(f'onnxruntime provider {provider} not available; '
                         f'have {avail}')
    chain = []
    if p == 'CoreMLExecutionProvider':
        # the (slow, one-off) compilation of the network is cached
        chain.append((p, {
            'ModelFormat': 'MLProgram',
            'MLComputeUnits': 'CPUAndGPU',
            'ModelCacheDirectory': str(
                config.Config.models_dir() / 'coreml-cache' / cache_name),
        }))
    elif p != 'CPUExecutionProvider':
        chain.append(p)
    chain.append('CPUExecutionProvider')
    return chain


class Sam2Predictor:
    """Same interface as the parts of SAM2ImagePredictor used by SMART."""

    def __init__(self, encoder_path, decoder_path, provider='cpu'):
        import onnxruntime as ort
        for p in (encoder_path, decoder_path):
            if not os.path.exists(p):
                raise ModelMissing(
                    f'SAM2 model not found: {p} (run with '
                    '--download-mask-model)')
        self._ort = ort
        self._enc_path = str(encoder_path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._so = so
        stem = Path(encoder_path).stem
        self.provider = provider
        self.enc = ort.InferenceSession(
            self._enc_path, so, providers=_providers(provider, stem))
        # the decoder is small: always on the CPU
        self.dec = ort.InferenceSession(
            str(decoder_path), so, providers=['CPUExecutionProvider'])
        self._feats = None
        self._size = None

    def set_image(self, image):
        """image: float32 HxWx3, RGB, 0..1"""
        h, w = image.shape[:2]
        x = preprocess(image)
        try:
            outs = self.enc.run(None, {'image': x})
        except Exception:
            if self.provider in (None, '', 'auto', 'cpu'):
                raise
            # an accelerator that cannot run the network: use the CPU
            self.provider = 'cpu'
            self.enc = self._ort.InferenceSession(
                self._enc_path, self._so, providers=['CPUExecutionProvider'])
            outs = self.enc.run(None, {'image': x})
        self._feats = outs      # high_res_feats_0, high_res_feats_1, embed
        self._size = (w, h)

    def predict(self, point_coords, point_labels, mask_input=None,
                multimask_output=False):
        """point_coords: Nx2 pixels (x, y) in the image; point_labels: N of
        1 (foreground) or 0 (background); mask_input: 1x256x256 logits of a
        previous prediction, or None.  Returns (masks 1xHxW bool, scores 1,
        low-res logits 1x256x256), like SAM2ImagePredictor.predict with
        multimask_output=False."""
        if self._feats is None:
            raise RuntimeError('set_image must be called first')
        if multimask_output:
            raise NotImplementedError('only single mask output')
        w, h = self._size
        pts = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        lbl = np.asarray(point_labels, dtype=np.float32).reshape(-1)
        if len(pts) == 0 or len(pts) != len(lbl):
            raise ValueError('need as many labels as (a non-zero number of) '
                             'points')
        pts = pts / np.array([w, h], dtype=np.float32) * SIZE
        if mask_input is None:
            mi = np.zeros((1, 1, LOW, LOW), dtype=np.float32)
            has = np.zeros(1, dtype=np.float32)
        else:
            mi = np.asarray(mask_input, dtype=np.float32).reshape(
                1, 1, LOW, LOW)
            has = np.ones(1, dtype=np.float32)
        f0, f1, emb = self._feats
        logits, iou = self.dec.run(None, {
            'image_embed': emb, 'high_res_feats_0': f0,
            'high_res_feats_1': f1,
            'point_coords': pts[None].astype(np.float32),
            'point_labels': lbl[None],
            'mask_input': mi, 'has_mask_input': has,
        })
        logits = logits[0]                       # 1x256x256
        mask = upsample_logits(logits[0], w, h) > 0.0
        return mask[None], iou[0], logits
