![logo](icons/app-icon.png)

# SMART

SMART is a simple GUI front-end to AI models, tailored to the
[ART](https://artraweditor.github.io/) raw image processor. It has two
modes:

- **AI masking** (the default): builds subject masks for an image using
  Segment Anything 2, to be used with ART.
- **AI healing**: removes unwanted elements by painting over them and
  filling the area with the LaMa inpainting model ("generative fill"),
  working on linear float32 TIFF files exported from ART without
  clipping.

The mode is selected with the `--mode` command-line switch: `--mode mask`
(the default) or `--mode heal`.

## Prerequisites

Make sure you have the following installed:

1. **Python 3**
2. **wxPython**: https://wxpython.org
3. **Pillow**: https://pillow.readthedocs.io/
4. **platformdirs**: http://pypi.org/project/platformdirs/
5. **NumPy**, **SciPy**, **tifffile** and **imagecodecs**
6. For AI masking: **SAM2**: https://github.com/facebookresearch/sam2
7. For AI healing: **onnxruntime** (default), or **PyTorch** (needed for
   Apple GPUs, and already installed together with SAM2)

All the dependencies can be installed with PIP, with the command
`pip install -r requirements.txt`.

## License

[GNU GPL](https://www.gnu.org/licenses/gpl-3.0.en.html)

The AI healing algorithm is a port of
[greycard](https://github.com/jessolmstead/greycard) (GPL-3.0). The LaMa
model it uses is released under the Apache-2.0 license.

## Setup

Run `SMART --init-config` from a terminal to create an initial
configuration file. The path to the file will be printed in output. Edit
the file with a text editor, and adapt the parameters to your setup. The
`device` parameter selects the device to use for computations: `"cuda"`
for a CUDA-capable GPU, `"mps"` for the GPU on an ARM Apple machine, or
`"cpu"` otherwise (this might be slow though).

Then download the model(s) for the mode(s) you want to use. They are put
in the `models/` directory.

### AI masking

Download one of the SAM 2 model checkpoints with
`SMART --download-mask-model QUALITY`, where `QUALITY` is one of `tiny`,
`small`, `base_plus` or `large` (larger is better but slower), or manually
from
https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints.
Then set the `model` parameter in the configuration file to the name of
the checkpoint to use.

### AI healing

Download the LaMa model with `SMART --download-heal-model`. This
selects the model according to `device`: the ONNX one by default, or the
PyTorch one with `"mps"` (to use the Apple GPU). You can choose
explicitly with `SMART --download-heal-model onnx` or
`SMART --download-heal-model torch`, in which case set also the
`heal_backend` parameter in the configuration file accordingly.

## Usage

By default the save dialog starts in the directory of the input image;
use `--outdir DIR` to start it in another directory.

### AI masking

Start SMART with `SMART [image]`. Build a mask for the image by adding
positive and negative points, by using *shift+left click* and
*shift+right click* respectively. The mask is built on the fly. When you
are happy, save the mask.

### AI healing

Start SMART with `SMART --mode heal [image]`. Paint over the areas to
heal with *shift+left drag* (*shift+right drag* erases). The mouse wheel
changes the brush radius, and *shift+wheel* the feather (`[` and `]` do
the same as the wheel, with and without shift); radius (in image pixels),
feather and opacity can also be set in the toolbar. Then press
*Cmd+Enter* on macOS or *Ctrl+Enter* elsewhere (or the *Heal* button) and
save the result. Zoom with the toolbar buttons
or *ctrl+plus* and *ctrl+minus*.

Linear float32 TIFF files are read and written without clipping, and
everything outside the healed area is left bit-exact. Other formats keep
their bit depth.
