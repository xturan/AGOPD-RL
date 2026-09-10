#!/usr/bin/env python3
"""Align a row of panels onto identical canvases so they render side-by-side aligned.

Takes N source PNGs, crops each to its non-white content bounding box, then
pastes every panel at the SAME left/top margin on a canvas whose size is the
union of the content sizes. All outputs therefore have identical pixel
dimensions and identical content origins -> equal displayed height in an HTML
2-column grid.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

MARGIN_X = 70
MARGIN_Y = 56

def _bbox(a: np.ndarray):
    nonwhite = (np.abs(a - 255).sum(axis=2) > 30)
    ys, xs = np.where(nonwhite)
    if len(ys) == 0:
        return (0, 0, a.shape[1], a.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

def align_row(src_files, dst_files, margin_x=MARGIN_X, margin_y=MARGIN_Y):
    imgs = [np.asarray(Image.open(f).convert('RGB')) for f in src_files]
    boxes = [_bbox(a) for a in imgs]
    # content may keep internal white gaps (between subplots / legend gutter): use
    # the raw non-white bbox but also allow padding so nothing is clipped.
    ws = [b[2]-b[0] for b in boxes]; hs = [b[3]-b[1] for b in boxes]
    W = max(ws) + 2*margin_x
    H = max(hs) + 2*margin_y
    for a, b, dst in zip(imgs, boxes, dst_files):
        x0, y0, x1, y1 = b
        crop = a[y0:y1, x0:x1]
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        canvas[margin_y:margin_y+crop.shape[0], margin_x:margin_x+crop.shape[1]] = crop
        Image.fromarray(canvas).save(dst)
        print('aligned', dst, '->', (W, H))

if __name__ == '__main__':
    import sys
    # usage: figcrop.py out_a out_b src_a src_b ... (pairs)
    files = sys.argv[1:]
    assert len(files) % 2 == 0
    for i in range(0, len(files), 2):
        align_row([files[i+2] if False else files[i]], [files[i+1]]) if False else None
