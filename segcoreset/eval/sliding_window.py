"""Sliding-window inference (mmseg convention, §2/§12.5), used for Cityscapes eval:
tile the image with overlapping crops, average overlapping logits, argmax once at the end.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sliding_window_inference(model, image: torch.Tensor, crop_size: tuple[int, int], stride: tuple[int, int],
                              num_classes: int, device: torch.device) -> torch.Tensor:
    """Predict a full-resolution image by averaging logits over overlapping crops.

    The image is zero-padded bottom/right up to at least `crop_size`, tiled with the given
    stride (with an extra final row/column of tiles flush with the bottom/right edge so
    every pixel is covered), each tile's logits are summed into a canvas, divided by the
    per-pixel tile count, cropped back to the original size, and argmaxed once.

    Args:
        model: A `SegmentationModel` on `device`, in eval mode.
        image: `[C, H, W]`, already normalized (CPU or `device`).
        crop_size: Tile size `(h, w)` — the training crop size.
        stride: Tile step `(h, w)`; smaller than `crop_size` gives overlap.
        num_classes: Number of output channels of `model`.
        device: Device for tiles and the logit accumulator.

    Returns:
        `[H, W]` predicted class ids (CPU, int64).
    """
    c, h, w = image.shape
    ch, cw = crop_size
    sh, sw = stride
    pad_h, pad_w = max(0, ch - h), max(0, cw - w)
    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    _, _, ph, pw = padded.shape

    ys = list(range(0, max(1, ph - ch + 1), sh))
    if ys[-1] != ph - ch:
        ys.append(max(0, ph - ch))
    xs = list(range(0, max(1, pw - cw + 1), sw))
    if xs[-1] != pw - cw:
        xs.append(max(0, pw - cw))

    logits_sum = torch.zeros(1, num_classes, ph, pw, device=device)
    count = torch.zeros(1, 1, ph, pw, device=device)

    for y in ys:
        for x in xs:
            crop = padded[:, :, y : y + ch, x : x + cw].to(device)
            logits_sum[:, :, y : y + ch, x : x + cw] += model(crop).float()
            count[:, :, y : y + ch, x : x + cw] += 1

    logits = (logits_sum / count.clamp(min=1))[:, :, :h, :w]
    return logits.argmax(dim=1).squeeze(0).cpu()
