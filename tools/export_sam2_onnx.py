#!/usr/bin/env python3
"""Export SAM2.1 to ONNX (encoder + single-mask decoder) for SMART.

This is a developer tool: it needs torch, onnx, onnxscript and sam2, none of
which are needed at runtime by SMART (which only uses onnxruntime).

Setup (separate venv, NOT the runtime one):
    uv venv --python 3.12 exportenv
    uv pip install --python exportenv/bin/python torch torchvision onnx \
        onnxscript onnxruntime numpy pillow hydra-core iopath tqdm \
        "git+https://github.com/facebookresearch/sam2.git"

Usage:
    python tools/export_sam2_onnx.py --size large \
        --checkpoint models/sam2.1_hiera_large.pt --out models/onnx \
        --image some.jpg

The wrappers are derived from vietanhdev/samexporter (Apache-2.0), with these
differences:
  * the decoder uses SAM2's single-mask path (multimask_output=False, with the
    stability-based fallback), like SAM2ImagePredictor.predict(...,
    multimask_output=False), which is what SMART uses;
  * the decoder returns the low-res (256x256) mask logits, so that they can be
    fed back as `mask_input` for iterative refinement;
  * the export is validated against the PyTorch SAM2ImagePredictor.

Decoder I/O:
  inputs : image_embed [1,256,64,64], high_res_feats_0 [1,32,256,256],
           high_res_feats_1 [1,64,128,128], point_coords [1,P,2] (pixels in
           1024x1024 space), point_labels [1,P] (1=fg, 0=bg, 2/3=box corners),
           mask_input [1,1,256,256], has_mask_input [1]
  outputs: low_res_masks [1,1,256,256] (logits, clamped to +-32),
           iou_predictions [1,1]
"""
import argparse
import hashlib
import pathlib

import numpy as np
import torch
from torch import nn

SIZES = {
    "tiny": "t",
    "small": "s",
    "base_plus": "b+",
    "large": "l",
}


class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, x):
        backbone_out = self.image_encoder(x)
        fpn = list(backbone_out["backbone_fpn"])
        fpn[0] = self.model.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.model.sam_mask_decoder.conv_s1(fpn[1])
        n = self.model.num_feature_levels
        feature_maps = fpn[-n:]
        pos = backbone_out["vision_pos_enc"][-n:]
        feat_sizes = [(p.shape[-2], p.shape[-1]) for p in pos]
        vision_feats = [f.flatten(2).permute(2, 0, 1) for f in feature_maps]
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        feats = [
            f.permute(1, 2, 0).reshape(1, -1, *s)
            for f, s in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        return feats[0], feats[1], feats[2]


class SAM2ImageDecoder(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.image_size = sam_model.image_size

    @torch.no_grad()
    def forward(self, image_embed, high_res_feats_0, high_res_feats_1,
                point_coords, point_labels, mask_input, has_mask_input):
        sparse = self._embed_points(point_coords, point_labels)
        dense = self._embed_masks(mask_input, has_mask_input)
        masks, iou, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )
        masks, iou = self.mask_decoder._dynamic_multimask_via_stability(
            masks, iou)
        return torch.clamp(masks, -32.0, 32.0), iou

    def _embed_points(self, coords, labels):
        pe = self.prompt_encoder
        coords = coords + 0.5
        pad_pt = torch.zeros_like(coords[:, :1, :])
        pad_lb = -torch.ones_like(labels[:, :1])
        coords = torch.cat([coords, pad_pt], dim=1)
        labels = torch.cat([labels, pad_lb], dim=1)
        coords = coords / self.image_size
        emb = pe.pe_layer._pe_encoding(coords)
        labels = labels.unsqueeze(-1).expand_as(emb)
        emb = emb * (labels != -1)
        emb = emb + pe.not_a_point_embed.weight * (labels == -1)
        for i in range(pe.num_point_embeddings):
            emb = emb + pe.point_embeddings[i].weight * (labels == i)
        return emb

    def _embed_masks(self, mask, has_mask):
        pe = self.prompt_encoder
        has_mask = has_mask.reshape(-1, 1, 1, 1)
        emb = has_mask * pe.mask_downscaling(mask)
        emb = emb + (1 - has_mask) * pe.no_mask_embed.weight.reshape(
            1, -1, 1, 1)
        return emb


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(size, checkpoint):
    from sam2.build_sam import build_sam2
    cfg = f"configs/sam2.1/sam2.1_hiera_{SIZES[size]}.yaml"
    return build_sam2(cfg, checkpoint, device="cpu").eval()


def export(model, out_dir, size, opset, dynamo):
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_path = out_dir / f"sam2.1_hiera_{size}_encoder.onnx"
    dec_path = out_dir / f"sam2.1_hiera_{size}_decoder.onnx"
    enc = SAM2ImageEncoder(model).eval()
    img = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        f0, f1, emb = enc(img)
    kw = dict(opset_version=opset, dynamo=dynamo)
    if not dynamo:
        kw["do_constant_folding"] = True
    torch.onnx.export(
        enc, (img,), str(enc_path),
        input_names=["image"],
        output_names=["high_res_feats_0", "high_res_feats_1", "image_embed"],
        **kw)
    print("saved", enc_path)

    dec = SAM2ImageDecoder(model).eval()
    # Realistic dummy prompts: 2 fg/bg points, so that both labels are traced
    coords = torch.tensor([[[300.0, 400.0], [700.0, 200.0]]])
    labels = torch.tensor([[1.0, 0.0]])
    mask_input = torch.randn(1, 1, 256, 256)
    has_mask = torch.tensor([1.0])
    dkw = dict(kw)
    if dynamo:
        dkw["dynamic_shapes"] = {
            "image_embed": None, "high_res_feats_0": None,
            "high_res_feats_1": None,
            "point_coords": {1: torch.export.Dim("num_points", min=1,
                                                 max=256)},
            "point_labels": {1: torch.export.Dim("num_points", min=1,
                                                 max=256)},
            "mask_input": None, "has_mask_input": None,
        }
    else:
        dkw["dynamic_axes"] = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
    torch.onnx.export(
        dec, (emb, f0, f1, coords, labels, mask_input, has_mask), str(dec_path),
        input_names=["image_embed", "high_res_feats_0", "high_res_feats_1",
                     "point_coords", "point_labels", "mask_input",
                     "has_mask_input"],
        output_names=["low_res_masks", "iou_predictions"],
        **dkw)
    print("saved", dec_path)
    return enc_path, dec_path


def preprocess(img_rgb_u8):
    """numpy-only version of what the runtime will do (see SAM2Transforms)."""
    from PIL import Image
    im = Image.fromarray(img_rgb_u8).resize((1024, 1024), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def validate(model, enc_path, dec_path, image_path):
    import onnxruntime as ort
    from PIL import Image
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    img = np.asarray(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    pred = SAM2ImagePredictor(model)
    pred.set_image(img)

    prov = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(str(enc_path), providers=prov)
    dec = ort.InferenceSession(str(dec_path), providers=prov)
    # Feed the very same preprocessed tensor to the ONNX encoder as torch used
    x = pred._transforms(img)[None].numpy()
    f0, f1, emb = enc.run(None, {"image": x})
    tf = pred._features
    print("encoder max abs diff: emb %.2e f0 %.2e f1 %.2e" % (
        np.abs(emb - tf["image_embed"].numpy()).max(),
        np.abs(f0 - tf["high_res_feats"][0].numpy()).max(),
        np.abs(f1 - tf["high_res_feats"][1].numpy()).max()))
    # numpy preprocessing difference (information only)
    xn = preprocess(img)
    print("numpy-vs-torch preprocess max abs diff: %.2e"
          % np.abs(xn - x).max())

    rng = np.random.default_rng(0)
    worst = 1.0
    for trial in range(5):
        n = int(rng.integers(1, 5))
        pts = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n)], 1)
        lbl = np.array([1] + list(rng.integers(0, 2, n - 1)))
        mask_in = None
        ref_logits = None
        o_logits = None
        for k in range(1, n + 1):  # incremental clicks, with refinement
            masks, _, logits = pred.predict(
                point_coords=pts[:k], point_labels=lbl[:k],
                mask_input=mask_in, multimask_output=False)
            mask_in = logits[:1]
            # onnx
            c = (pts[:k] / np.array([W, H]) * 1024)[None].astype(np.float32)
            has = np.array([0.0 if o_logits is None else 1.0], np.float32)
            mi = (o_logits if o_logits is not None
                  else np.zeros((1, 1, 256, 256), np.float32))
            o_logits, o_iou = dec.run(None, {
                "image_embed": emb, "high_res_feats_0": f0,
                "high_res_feats_1": f1, "point_coords": c,
                "point_labels": lbl[:k][None].astype(np.float32),
                "mask_input": mi, "has_mask_input": has})
            d = np.abs(o_logits[0, 0] - logits[0]).max()
            # compare binary masks at full res via bilinear upsample
            t = torch.from_numpy(o_logits)
            up = torch.nn.functional.interpolate(
                t, (H, W), mode="bilinear", align_corners=False)[0, 0] > 0
            inter = np.logical_and(up.numpy(), masks[0] > 0).sum()
            union = np.logical_or(up.numpy(), masks[0] > 0).sum()
            iou = inter / union if union else 1.0
            worst = min(worst, iou)
            print(f"trial {trial} clicks {k}: logits max abs diff {d:.3e}, "
                  f"mask IoU {iou:.5f}")
    print("worst IoU:", worst)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--size", choices=tuple(SIZES), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--dynamo", action="store_true",
                    help="use the torch.export based exporter")
    ap.add_argument("--image", help="image used to validate the export")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    model = build(args.size, args.checkpoint)
    out = pathlib.Path(args.out)
    if args.skip_export:
        enc_path = out / f"sam2.1_hiera_{args.size}_encoder.onnx"
        dec_path = out / f"sam2.1_hiera_{args.size}_decoder.onnx"
    else:
        enc_path, dec_path = export(model, out, args.size, args.opset,
                                    args.dynamo)
    for p in (enc_path, dec_path):
        print(f"{sha256(p)}  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    if args.image:
        validate(model, enc_path, dec_path, args.image)


if __name__ == "__main__":
    main()
