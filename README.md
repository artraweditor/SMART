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

- **Raw denoising**: removes noise from a mosaiced (Bayer) DNG with a
  learned raw-domain denoiser, with a before/after preview, either from
  the GUI or from the command line.

The mode is selected with the `--mode` command-line switch: `--mode mask`,
`--mode heal` or `--mode denoise`. If the switch is not given, the GUI asks
which mode to use when it starts.

## Prerequisites

Make sure you have the following installed:

1. **Python 3**
2. **wxPython**: https://wxpython.org
3. **Pillow**: https://pillow.readthedocs.io/
4. **platformdirs**: http://pypi.org/project/platformdirs/
5. **NumPy**, **SciPy**, **tifffile** and **imagecodecs**
6. For AI masking: **SAM2**: https://github.com/facebookresearch/sam2
7. For AI healing and raw denoising: **onnxruntime** (default), or
   **PyTorch** (needed for Apple GPUs, and already installed together
   with SAM2)

All the dependencies can be installed with PIP, with the command
`pip install -r requirements.txt`.

## License

[GNU GPL](https://www.gnu.org/licenses/gpl-3.0.en.html)

The AI healing and the raw denoising code are ported from
[greycard](https://github.com/jessolmstead/greycard) (GPL-3.0). 
The LaMa model used for AI healing is released under the Apache-2.0 license. 
The raw denoising models are
[those released by the greycard author](https://huggingface.co/jessolmstead/greycard-denoise).

## Setup

Run `SMART --init-config` from a terminal to create an initial
configuration file. The path to the file will be printed in output. Edit
the file with a text editor, and adapt the parameters to your setup. The
`device` parameter selects the device to use for computations: `"cuda"`
for a CUDA-capable GPU, `"mps"` for the GPU on an ARM Apple machine, or
`"cpu"` otherwise (this might be slow though).

Then download the model(s) for the mode(s) you want to use. They are put
in the `models/` directory. (In the GUI, if the model of the selected mode
is missing, SMART offers to download it when it starts, or, in denoise mode,
when you press the Denoise button.)

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

### Raw denoising

Download the denoising model(s) with `SMART --download-denoise-model`
(the configured one, `balanced` by default), `SMART --download-denoise-model
best`, or `SMART --download-denoise-model all` (`fast`, `balanced` and
`best` are 4.9, 9 and 20 MB; larger is better but slower). They are put in
the `models/` directory. The `denoise_*` parameters of the configuration
file set the default model, backend (`onnxruntime` or `torch`), provider
or device (`auto`), tile size and margin, and DNG output depth.

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

### Raw denoising

The input must be a mosaiced DNG as written by
`ART-cli --raw-preprocess=mosaic` (so that ART does the raw
preprocessing: dead pixels, flat field, white balance baking, ...):
uncompressed, 16-bit integer or 32-bit float, one sample per pixel,
Bayer pattern (X-Trans is not supported by the model). The DNG reader is
built on tifffile and nothing else (no rawpy/LibRaw). ART writes
no black level (it is 0) and the data are already cropped to the active
area, so `BlackLevel` and `ActiveArea` tags are ignored.

Start SMART with `SMART --mode denoise [file.dng]`, choose the model in the
toolbar and press *Denoise* (or *Cmd/Ctrl+Enter*). The *View* selector
shows the before/after split view (drag the divider), or either image alone;
the wheel zooms around the pointer and dragging pans. The preview is the raw
data rendered linearly: the as-shot white balance and camera matrix to
sRGB, then only the sRGB encoding and the display profile (no tone curve or
exposure change). The "before" image uses plain bilinear demosaicing; the
network demosaics and denoises at once. Save the result as a float32 linear
TIFF (camera RGB, no white balance) or as a linear DNG (16-bit or float,
with camera matrix, as-shot neutral and metadata).

Without the GUI:

    SMART --mode denoise --cli photo.dng -o photo.tif
    SMART --mode denoise --cli photo.dng -m best --provider cpu
    SMART --mode denoise --cli photo.dng -o photo-dn.dng --dng-bits 32
    SMART --mode denoise --cli photo.dng --backend torch

The default output is `photo-denoised.tif`. Other options: `--model-file`,
`--noise aR,bR,aG,bG,aB,bB`, `-f tiff|dng`, `--wb camera` (TIFF only),
`--no-rotate` (TIFF only; the DNG keeps the orientation flag), `--tile`,
`--margin`, `-v`, and `--progress` (integer percentages on stdout).
Options not given default to the configuration file.
