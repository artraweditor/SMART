#!/usr/bin/env python3
"""Make a CoreML-friendly version of the dynamic LaMa ONNX model.

The onnxruntime CoreML provider cannot run big-lama-dynamic-fp32.onnx as is:
it rejects a few things in the graph, which makes it fail or split the model
into dozens of CPU/GPU partitions:
  * Gather of a constant index on the real/imaginary axis (the FFT, which
    the model implements with matrix products) -> Slice + Squeeze;
  * Neg -> Mul by -1;
  * ConvTranspose with output_padding (asymmetric pads) -> ConvTranspose
    without padding, followed by a Slice.
The result is the same (to about 1e-5) and the model stays dynamic and works
on every provider.  SMART gives the CoreML provider one static session per
window size (free dimension overrides of '8*hk' and '8*wk').

Needs only onnx (and onnxruntime to check the result):

    python tools/make_lama_coreml.py models/big-lama-dynamic-fp32.onnx \
        --out models/big-lama-dynamic-coreml.onnx --check
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite_gathers(graph):
    consts = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == 'Constant':
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    new_nodes, extra = [], []
    count = 0
    for n in graph.node:
        idx = consts.get(n.input[1]) if n.op_type == 'Gather' else None
        if idx is None or np.ndim(idx) != 0:
            new_nodes.append(n)
            continue
        axis = next((a.i for a in n.attribute if a.name == 'axis'), 0)
        i = int(idx)
        tag = f'{n.output[0]}_g{count}'
        count += 1
        for suffix, val in (('starts', [i]), ('ends', [i + 1]),
                            ('axes', [axis])):
            extra.append(numpy_helper.from_array(
                np.array(val, np.int64), f'{tag}_{suffix}'))
        mid = tag + '_sl'
        new_nodes.append(helper.make_node(
            'Slice', [n.input[0], f'{tag}_starts', f'{tag}_ends',
                      f'{tag}_axes'], [mid], name=tag + '_slice'))
        new_nodes.append(helper.make_node(
            'Squeeze', [mid, f'{tag}_axes'], [n.output[0]],
            name=tag + '_squeeze'))
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(extra)
    return count


def rewrite_neg(graph):
    """Neg(x) -> Mul(x, -1)  (Neg is not supported by the CoreML provider)."""
    one = numpy_helper.from_array(np.array(-1.0, np.float32), 'minus_one')
    graph.initializer.append(one)
    count = 0
    for n in graph.node:
        if n.op_type == 'Neg':
            n.op_type = 'Mul'
            n.input.append('minus_one')
            count += 1
    return count


def rewrite_convtranspose(graph):
    """ConvTranspose with output_padding (or other asymmetric pads) -> the
    same without any padding, followed by a Slice: CoreML only takes equal
    pads on all sides and no output_padding."""
    new_nodes, extra, count = [], [], 0
    for n in graph.node:
        attrs = {a.name: a for a in n.attribute}
        op = attrs.get('output_padding')
        if n.op_type != 'ConvTranspose' or op is None or not any(op.ints):
            new_nodes.append(n)
            continue
        pads = list(attrs['pads'].ints)
        r = len(op.ints)
        begin = pads[:r]
        # the output is (full - begin - end + output_padding): the same as
        # cropping the unpadded output by `begin` on the left and by
        # `end - output_padding` on the right
        end = [pads[r + k] - op.ints[k] for k in range(r)]
        assert all(e >= 0 for e in end), 'cannot fold output_padding'
        del attrs['pads'].ints[:]
        attrs['pads'].ints.extend([0] * (2 * r))
        del op.ints[:]
        op.ints.extend([0] * r)
        tag = f'{n.output[0]}_ct{count}'
        count += 1
        full = tag + '_full'
        out = n.output[0]
        n.output[0] = full
        big = np.iinfo(np.int64).max
        for suffix, val in (
                ('starts', begin),
                ('ends', [big if e == 0 else -e for e in end]),
                ('axes', [2 + k for k in range(r)])):
            extra.append(numpy_helper.from_array(
                np.array(val, np.int64), f'{tag}_{suffix}'))
        new_nodes.append(n)
        new_nodes.append(helper.make_node(
            'Slice', [full, f'{tag}_starts', f'{tag}_ends', f'{tag}_axes'],
            [out], name=tag + '_crop'))
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(extra)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('--size', type=int, default=512,
                    help='side of the square window used by --check')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true',
                    help='compare with the original on the CPU, and time '
                    'the CoreML provider')
    args = ap.parse_args()
    assert args.size % 8 == 0
    m = onnx.load(args.model)
    print('gathers rewritten:', rewrite_gathers(m.graph))
    print('negations rewritten:', rewrite_neg(m.graph))
    print('transposed convolutions fixed:', rewrite_convtranspose(m.graph))
    onnx.checker.check_model(m)
    onnx.save(m, args.out)
    print('saved', args.out)
    if args.check:
        check(args.model, args.out, args.size)


def check(orig, new, n):
    import time
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    mask = np.zeros((1, 1, n, n), np.float32)
    mask[:, :, n // 3:n // 2, n // 4:n // 2] = 1
    img = (rng.random((1, 3, n, n)).astype(np.float32)) * (1 - mask)
    feed = {'image': img, 'mask': mask}
    ref = ort.InferenceSession(orig, providers=['CPUExecutionProvider']) \
        .run(['output'], feed)[0]
    cases = [('cpu', ['CPUExecutionProvider'], {})]
    if 'CoreMLExecutionProvider' in ort.get_available_providers():
        cases.append(('coreml', [('CoreMLExecutionProvider', {
            'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}),
            'CPUExecutionProvider'], {'8*hk': n, '8*wk': n}))
    for name, prov, free in cases:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        for k, v in free.items():
            so.add_free_dimension_override_by_name(k, v)
        t = time.time()
        s = ort.InferenceSession(new, so, providers=prov)
        load = time.time() - t
        out = s.run(['output'], feed)[0]
        t = time.time()
        for _ in range(3):
            s.run(['output'], feed)
        run = (time.time() - t) / 3
        print(f'{name}: load {load:.1f}s, run {run:.3f}s, max diff vs '
              f'original {np.abs(out - ref).max():.2e}')


if __name__ == '__main__':
    main()
