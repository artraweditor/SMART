import numpy as np
from PIL import Image, ImageCms


class BaseEngine:
    """State and display handling shared by the masking and healing engines."""

    def __init__(self, conf):
        self.conf = conf
        self.image_filename = None
        self.displayed_image = None
        self.saved = True
        self.size = -1, -1
        self.srgb_profile = ImageCms.createProfile('sRGB')
        if self.conf.display_icc_profile:
            display_profile = ImageCms.ImageCmsProfile(
                self.conf.display_icc_profile)
        else:
            display_profile = ImageCms.get_display_profile()
        if display_profile is not None:
            self.display_xform = ImageCms.buildTransform(
                self.srgb_profile, display_profile, "RGB", "RGB")
        else:
            self.display_xform = None

    def get_size(self):
        return self.size

    def busy_now(self):
        """True while a background operation is running."""
        return False

    def to_display(self, img):
        """sRGB float image in 0..1 -> uint8 in the display's color space."""
        ret = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        if self.display_xform is not None:
            src = Image.fromarray(ret)
            ImageCms.applyTransform(src, self.display_xform, True)
            ret = np.array(src)
        return ret
