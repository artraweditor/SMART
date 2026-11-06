from pathlib import Path
import io
import os
import json
import subprocess
import hashlib

import numpy as np
from PIL import Image, ImageCms
import config
from engine_base import BaseEngine


def checksum(filename):
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class AIMaskingEngine(BaseEngine):
    def __init__(self, conf):
        # SAM2 is only needed in masking mode
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        super().__init__(conf)
        sam2_checkpoint = self.conf.get_model_file()
        model_cfg_dir, model_cfg = self.conf.get_model_config()
        with hydra.initialize_config_dir(
                version_base=None,
                config_dir=str(Path(model_cfg_dir).resolve())):
            self.sam2_model = build_sam2(model_cfg,
                                         sam2_checkpoint,
                                         device=self.conf.device)
            self.sam2_model.eval()
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.image = None
        self.mask = None
        self.mask_input = None
        self.points = []
        self.labels = []

    def load_image(self, filename):
        self.reset(True)
        def doit(fn):
            img = Image.open(fn).convert('RGB')
            icc = img.info.get('icc_profile')
            if icc is not None:
                try:
                    iprof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                    xform = ImageCms.buildTransform(
                        iprof, self.srgb_profile, "RGB", "RGB")
                    ImageCms.applyTransform(img, xform, True)
                except:
                    import traceback
                    traceback.print_exc()
            self.size = img.size
            self.image = np.array(img).astype(np.float32) / 255.0
            self.mask = None
            self.displayed_image = self.to_display(self.image)
            self.predictor.set_image(self.image)
            self.image_filename = os.path.abspath(fn)
        if self.conf.exiftool:
            try:
                res = subprocess.run(
                    [self.conf.exiftool, '-json', '-Smart_mask_data', filename],
                    encoding='utf-8', capture_output=True, check=True)
                md = json.loads(res.stdout)[0]
                info = json.loads(md['Smart_mask_data'])
                if info is not None and isinstance(info, dict):
                    fn = info.get('image', '')
                    cs = info.get('sha256sum')
                    ps = info.get('points')
                    ls = info.get('labels')
                    if os.path.exists(fn) and cs is not None \
                       and ps is not None and ls is not None \
                       and len(ps) == len(ls) and checksum(fn) == cs:
                        doit(fn)
                        self.points = ps
                        self.labels = ls
                        self.predict()
                        return
            except Exception:
                pass
        doit(filename)

    def add_point(self, point, is_positive):
        self.saved = False
        self.points.append(point)
        self.labels.append(int(is_positive))
        self.predict()

    def predict(self):
        if self.image is not None:
            masks, scores, logits = self.predictor.predict(
                point_coords=self.points,
                point_labels=self.labels,
                mask_input=self.mask_input,
                multimask_output=False
            )
            self.mask = masks[0]
            self.mask_input = logits[0, :, :][None, :, :]
            c = np.array(self.conf.mask_color, dtype=np.float32) / 255.0
            cm = np.zeros_like(self.image)[:,:] = c
            zm = np.zeros_like(self.image)
            m = np.repeat(np.expand_dims(self.mask, axis=2), 3, axis=2)
            m = np.where(m > 0, cm, zm)
            s = 0.5
            w = [0.2225045 * (1-s),  0.7168786 * (1-s),  0.0606169 * (1-s)]
            desat = np.array([
                [w[0] + s, w[0], w[0]],
                [w[1], w[1] + s, w[1]],
                [w[2], w[2], w[2] + s]
            ], dtype=np.float32).transpose()
            img = (desat @ self.image.reshape(-1, 3).transpose()).\
                transpose().reshape(self.image.shape)
            self.displayed_image = self.to_display(np.fmin(img + m * 0.5, 1.0))

    def reset(self, clear_image):
        self.saved = True
        self.points = []
        self.labels = []
        self.mask = None
        self.mask_input = None
        if clear_image:
            self.image = None
            self.displayed_image = None
        else:
            self.displayed_image = self.to_display(self.image)

    def undo_last(self):
        if self.points:
            self.points.pop()
            self.labels.pop()
            self.saved = not self.points
            self.predict()

    def save_mask(self, filename):
        if self.image is None:
            raise Exception("no image loaded")
        if self.mask is not None:
            mask = (self.mask * 255).astype(np.uint8)
        else:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        img = Image.fromarray(mask)
        img.save(filename)
        data = {
            'image' : self.image_filename,
            'sha256sum' : checksum(self.image_filename),
            'points' : self.points,
            'labels' : self.labels,
        }
        if self.conf.exiftool:
            subprocess.run([
                self.conf.exiftool,
                '-config',
                str(config.resource_dir() / 'data' / 'exiftool.config'),
                '-overwrite_original',
                '-Smart_mask_data=' + json.dumps(data),
                filename], capture_output=True)
        self.saved = True

# end of class AIMaskingEngine
