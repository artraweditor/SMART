<img src="icons/app-icon-1024.png" alt="logo" width="160">

# SMART

SMART is a simple GUI front-end to AI models, tailored to the
[ART](https://artraweditor.github.io/) raw image processor. It has three
modes:

- **AI masking** (the default): builds subject masks for an image using
  Segment Anything 2, to be used with ART.

- **AI healing**: removes unwanted elements by painting over them and
  filling the area with the LaMa inpainting model ("generative fill"),
  working on linear float32 TIFF files exported from ART without
  clipping.

- **AI denoising**: removes noise from a mosaiced (Bayer) or a linear
  (demosaiced) raw DNG with a learned denoiser, with a before/after
  preview, either from the GUI or from the command line.

The mode is selected with the `--mode` command-line switch: `--mode mask`,
`--mode heal` or `--mode denoise`. If the switch is not given, the GUI asks
which mode to use when it starts.

## Prerequisites

SMART needs Python 3 and the following packages:

- [wxPython](https://wxpython.org)
- [onnxruntime](https://onnxruntime.ai/), which runs the neural networks of
  all three modes
- [Pillow](https://pillow.readthedocs.io/)
- [platformdirs](https://pypi.org/project/platformdirs/)
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/),
  [tifffile](https://pypi.org/project/tifffile/) and
  [imagecodecs](https://pypi.org/project/imagecodecs/)

You can install them all with `pip install -r requirements.txt`.

Optionally, [ExifTool](https://exiftool.org/) lets AI denoising copy the
exposure and lens metadata of the input into the output DNG.

PyTorch is not needed: it is only required by `tools/export_sam2_onnx.py`,
which creates the SAM2 models (see `tools/requirements-export.txt`).

## License

[GNU GPL](https://www.gnu.org/licenses/gpl-3.0.en.html)

The AI healing and the AI denoising code are ported from
[greycard](https://github.com/jessolmstead/greycard) (GPL-3.0). 
The LaMa model used for AI healing is released under the Apache-2.0 license. 
The denoising models for mosaiced DNGs are
[those released by the greycard author](https://huggingface.co/jessolmstead/greycard-denoise).
For linear DNGs, the TreeNet models of
[RawForge](https://github.com/rymuelle/RawForge) (MIT) are used, as
repackaged in [rawlab-models](https://huggingface.co/AyedaOk/rawlab-models).

## Setup

Run `SMART --init-config` from a terminal to create an initial
configuration file. The path to the file will be printed in output. Edit
the file with a text editor, and adapt the parameters to your setup. The
`mask_provider`, `heal_providers` and `denoise_provider` parameters select
the onnxruntime execution providers used by the three modes (e.g.
`"cuda"`, `"coreml"`, `"directml"`); the default is the CPU for masking and
healing, and the best available one for denoising.

Then download the model(s) for the mode(s) you want to use. They are put
in the `models/` directory. (In the GUI, if the model of the selected mode
is missing, SMART offers to download it when it starts, or, in denoise mode,
when you press the Denoise button.)

### AI masking

Download one of the SAM 2.1 models with
`SMART --download-mask-model QUALITY`, where `QUALITY` is one of `tiny`,
`small`, `base_plus` or `large` (larger is better but slower), then set the
`model` parameter in the configuration file to the same name. The models
are ONNX versions of Meta's SAM 2.1 checkpoints (an image encoder and a
mask decoder, 126, 155, 294 and 869 MB in total), made with
`tools/export_sam2_onnx.py`, which also checks them against the PyTorch
implementation. On the CPU the encoder takes a few seconds per image
(2.6 s for `large` on an Apple M-series machine) and each click is almost
instantaneous. With `"mask_provider": "coreml"` the encoder is faster
(1.4 s) but the first start compiles the network, which takes several minutes.

### AI healing

Download the LaMa model with `SMART --download-heal-model`. On a Mac it runs
on the GPU through CoreML by default (about 3 times faster than the CPU for
the larger windows); elsewhere on the CPU, or on the providers listed in
`heal_providers` (e.g. `["CUDAExecutionProvider"]`; `["cpu"]` to force the
CPU). CoreML needs a fixed window size: SMART rounds windows up to 256, 384,
512, 768 or 1024 pixels (smaller ones run on the CPU). The first time each
size is used it takes a second or two to compile the network, and the result
is cached in `models/coreml-cache` (about 0.4 GB for each size).

The model is the one of [greycard](https://huggingface.co/jessolmstead/greycard-lama),
with a few operations rewritten for CoreML by `tools/make_lama_coreml.py`.

### AI denoising

Download the denoising models with `SMART --download-denoise-model`
(the configured one, `medium` by default), `SMART --download-denoise-model
large`, or `SMART --download-denoise-model all`. There are three sizes,
`small`, `medium` and `large` (4.9, 9 and 20 MB for mosaiced DNGs; larger is
better but slower). Each one also has a network for linear DNGs (the TreeNet
`light`, standard and `heavy` models: 1.5, 10 and 22 MB), which is downloaded
together with it. In the GUI, a missing model is offered for download when
needed. The models are put in the `models/` directory.

The `denoise_*` parameters of the configuration file set the default model,
onnxruntime provider (`auto`), tile size and margin (for mosaiced DNGs), and
DNG output depth.

## Usage

By default the save dialog starts in the directory of the input image;
use `--outdir DIR` to start it in another directory.

### Integration with ART

`--save-as FILE` (all modes) sets the default save name: the Save button
writes to `FILE` straight away, without the file dialog (which is only shown
if `FILE` already exists), and the Open button is hidden. The extension
selects the output format in denoise mode.

An ART user command can thus have SMART save to a temporary file, and
move it next to the input image (with the sidecar) when SMART exits and the
file was written.

For more detailed instructions on how to integrate SMART with ART, see
<https://artraweditor.github.io/AItools>.

### AI masking

Start SMART with `SMART --mode mask [image]`.
 Build a mask for the image by adding
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

### AI denoising

**Input.** A DNG written by `ART-cli --raw-preprocess=mosaic` or
`ART-cli --raw-preprocess=demosaic`, so that ART does the raw preprocessing
(dead pixels, flat field, white balance baking, ...). It must be
uncompressed, 16-bit integer or 32-bit float, and either a mosaic with one
sample per pixel (Bayer pattern; X-Trans is not supported) or a linear image
with three samples per pixel. The model choices (`small`, `medium`, `large`)
are the same for both: the greycard denoiser (which demosaics and denoises
at once) is used for mosaics, and TreeNet for linear files.

The black level (with its repeat pattern and the per-row and per-column
deltas) is subtracted and the result is divided by `WhiteLevel` minus black,
as in the DNG specification; values below black are set to 0. The
`ActiveArea` of a mosaic, if any, is cropped.

**GUI.** Start SMART with `SMART --mode denoise [file.dng]`, choose the model
in the toolbar and press *Denoise* (or *Cmd/Ctrl+Enter*). The *View*
selector shows the before/after split view (drag the divider), or either
image alone; the wheel zooms around the pointer and dragging pans. The
preview is the raw data rendered linearly (as-shot white balance and camera
matrix to sRGB, then only the sRGB encoding and the display profile: no tone
curve or exposure change). The "before" image of a mosaic uses plain
bilinear demosaicing, since the network demosaics and denoises at once; a
linear DNG is shown as it is.

**Output.** Save the result as a float32 linear TIFF (camera RGB, no white
balance) or as a linear DNG (16-bit or float, with camera matrix, as-shot
neutral and metadata). The DNG keeps the orientation flag and the lens
correction opcodes (distortion and vignetting) of the input, if present, for
example those created with `ART-cli --dng-lens-opcodes`.

**Command line.**

    SMART --mode denoise --cli photo.dng -o photo.tif
    SMART --mode denoise --cli linear.dng -m large -o linear-dn.dng
    SMART --mode denoise --cli photo.dng -m large --provider cpu
    SMART --mode denoise --cli photo.dng -o photo-dn.dng --dng-bits 32
    SMART --mode denoise --cli photo.dng --provider coreml

The default output is `photo-denoised.tif`. Other options:

- `--model-file`: use this ONNX file instead of a model
- `--noise aR,bR,aG,bG,aB,bB`: noise model, instead of estimating it
  (mosaics)
- `--iso`: the ISO of linear DNGs
- `-f tiff|dng`: output format (default from the `-o` extension)
- `--wb camera`, `--no-rotate`: TIFF output only
- `--tile`, `--margin`: tiling
- `-v`, and `--progress` (integer percentages on stdout)

Options not given default to the configuration file.
