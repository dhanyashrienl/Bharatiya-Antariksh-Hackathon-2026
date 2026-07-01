"""
inference_stitch.py
-------------------
Seamless full-image inference using Gaussian-weighted patch blending.

Solves the patch boundary artifact problem: each patch's contribution
is weighted by a 2D Gaussian, so patch centers dominate and edges
fade out. Overlapping patches blend smoothly without visible seams.

Usage:
    from scripts.inference_stitch import stitch_inference, to_bgr_deliverable
    output_rgb = stitch_inference(tir_image, model, patch_size=256, stride=128)
    # MANDATORY: convert to BGR before writing the submission TIFF.
    deliverable_bgr = to_bgr_deliverable(output_rgb)   # BGR, uint16 by default

Stride < patch_size creates overlap. Recommended: stride = patch_size // 2

BAND-ORDER CONTRACT (I-3):
    stitch_inference() returns an RGB array (intermediate, for display/debug).
    The FINAL submission TIFF MUST be BGR-ordered (mandatory eval requirement).
    Always pass the stitched output through to_bgr_deliverable() before writing
    the deliverable. Do NOT write the RGB array directly to the submission TIFF.
"""

import numpy as np
import torch
import cv2
import logging

logger = logging.getLogger(__name__)


def make_gaussian_weight_map(patch_size: int, sigma_factor: float = 0.35) -> np.ndarray:
    """
    Creates a 2D Gaussian weight map for a square patch.

    The center pixel gets weight ~1.0, edges get weight ~0.0.
    This means when two overlapping patches are blended, the center
    of each patch "wins" over its edges — eliminating hard seams.

    Args:
        patch_size: Height and width of the square patch (e.g., 256).
        sigma_factor: Controls Gaussian spread. 0.35 means sigma = 0.35 * patch_size.
                      Smaller = sharper falloff at edges.

    Returns:
        weight_map: (patch_size, patch_size) float32 array, values in [0, 1].
    """
    sigma = sigma_factor * patch_size
    center = patch_size / 2.0

    # Create coordinate grids
    x = np.arange(patch_size) - center
    y = np.arange(patch_size) - center
    xx, yy = np.meshgrid(x, y)

    # 2D isotropic Gaussian: G(x,y) = exp(-(x^2 + y^2) / (2 * sigma^2))
    weight_map = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return weight_map.astype(np.float32)


def stitch_inference(
    tir_image: np.ndarray,
    model,
    patch_size: int = 256,
    stride: int = 128,
    device: str = 'cuda',
    out_channels: int = 3,
) -> np.ndarray:
    """
    Runs model inference on a full TIR image using overlapping patches
    with Gaussian-weighted blending. Eliminates patch seam artifacts.

    Args:
        tir_image   : Input TIR array, shape (H, W) or (1, H, W), float32, PN-calibrated.
        model       : Trained Pix2Pix generator (or any model) in eval mode.
                      Expected input:  (1, 1, patch_size, patch_size) tensor
                      Expected output: (1, out_channels, patch_size, patch_size) tensor
        patch_size  : Size of square patches the model was trained on (default 256).
        stride      : Step between patches. stride < patch_size creates overlap.
                      Recommended: stride = patch_size // 2 (50% overlap).
        device      : 'cuda' or 'cpu'.
        out_channels: Number of output channels (3 for RGB).

    Returns:
        output_image: (H, W, out_channels) uint8 RGB array, seamlessly stitched.
    """
    if tir_image.ndim == 3:
        # (1, H, W) -> (H, W)
        tir_image = tir_image.squeeze(0)

    H0, W0 = tir_image.shape

    # I-11: edge guard. The sliding window and the W-patch_size/H-patch_size edge
    # offsets break (negative slices) when the image is smaller than a patch.
    # Pad the image up to at least patch_size in each dim, run inference on the
    # padded image, then crop the result back to the original (H0, W0).
    pad_h = max(0, patch_size - H0)
    pad_w = max(0, patch_size - W0)
    if pad_h or pad_w:
        logger.warning(
            f"Image {H0}x{W0} smaller than patch {patch_size}; "
            f"padding by ({pad_h}, {pad_w}) and cropping result back."
        )
        tir_image = np.pad(tir_image, ((0, pad_h), (0, pad_w)), mode='reflect')

    H, W = tir_image.shape
    logger.info(f"Full image size: {H}x{W}, patch={patch_size}, stride={stride}")

    # ── Accumulators ──────────────────────────────────────────────────────────
    # output_acc  : accumulates weighted RGB predictions
    # weight_acc  : accumulates the Gaussian weights (for normalization)
    output_acc = np.zeros((H, W, out_channels), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)

    gauss = make_gaussian_weight_map(patch_size)           # (P, P)
    gauss_3ch = gauss[:, :, np.newaxis]                    # (P, P, 1) for broadcasting

    model.eval()
    patch_count = 0

    # Compute global normalization statistics from valid (non-zero) pixels once.
    # Per-patch local normalization destroys absolute temperature information —
    # a 305K water patch and a 320K soil patch would both map to identical [-1,1],
    # making the physics loss (Stefan-Boltzmann ΔT) meaningless.
    valid_pixels = tir_image[tir_image > 0] if tir_image.min() >= 0 else tir_image.flatten()
    if valid_pixels.size == 0:
        valid_pixels = tir_image.flatten()
    # I-10: use robust percentiles instead of raw min()/max(). A single saturated
    # pixel (e.g. a 65535 hot spot or a 0 nodata cell) otherwise compresses the
    # entire dynamic range. The global-stats design (one range for the whole
    # image, not per-patch) is intentional and preserved.
    G_MIN = float(np.percentile(valid_pixels, 1))
    G_MAX = float(np.percentile(valid_pixels, 99))
    if G_MAX - G_MIN < 1e-6:
        G_MAX = G_MIN + 1.0
    logger.info(f"Global normalization range (robust p1/p99): [{G_MIN:.1f}, {G_MAX:.1f}]")

    # I-11: build SORTED UNIQUE offset sets so the right/bottom edge patch
    # (offset = W-patch_size / H-patch_size) is included exactly once. The old
    # code ran explicit edge passes that re-accumulated an exact-fit edge patch
    # a second time (double Gaussian weight). Using a set de-duplicates the case
    # where (W-patch_size) is already a multiple of stride.
    x_offsets = sorted(set(list(range(0, W - patch_size + 1, stride)) + [W - patch_size]))
    y_offsets = sorted(set(list(range(0, H - patch_size + 1, stride)) + [H - patch_size]))

    with torch.no_grad():
        for y in y_offsets:
            for x in x_offsets:

                # ── Extract patch ─────────────────────────────────────────────
                patch = tir_image[y:y + patch_size, x:x + patch_size]

                # Skip blank/nodata patches
                if patch.max() - patch.min() < 1e-6:
                    continue

                # Normalize using GLOBAL image statistics (not per-patch local min/max).
                # Clip to [-1, 1]: with robust p1/p99 stats, pixels outside the
                # percentile band would otherwise map beyond the model's input range.
                patch_norm = 2.0 * (patch - G_MIN) / (G_MAX - G_MIN) - 1.0
                patch_norm = np.clip(patch_norm, -1.0, 1.0)

                # ── Model forward pass ────────────────────────────────────────
                tensor = torch.from_numpy(patch_norm).float()
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, P, P)

                pred = model(tensor)                                   # (1, C, P, P)
                pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (P, P, C)

                # Rescale model output from [-1, 1] → [0, 255]
                pred_np = (pred_np + 1.0) / 2.0 * 255.0
                pred_np = np.clip(pred_np, 0, 255)

                # ── Gaussian-weighted accumulation ────────────────────────────
                output_acc[y:y + patch_size, x:x + patch_size] += pred_np * gauss_3ch
                weight_acc[y:y + patch_size, x:x + patch_size] += gauss

                patch_count += 1

    logger.info(f"Processed {patch_count} patches.")

    # ── Normalize: divide accumulated output by accumulated weights ───────────
    # Pixels at the center of many patches get large weight_acc values,
    # but division normalizes them back. Edge pixels (low weight_acc) also normalize.
    weight_acc = np.maximum(weight_acc, 1e-8)[:, :, np.newaxis]  # avoid div/0
    output_final = output_acc / weight_acc
    output_final = np.clip(output_final, 0, 255).astype(np.uint8)

    # I-11: crop back to the original size if the image was padded up to patch_size.
    output_final = output_final[:H0, :W0, :]

    return output_final  # (H0, W0, 3) RGB uint8 (intermediate; see to_bgr_deliverable)


def to_bgr_deliverable(rgb_image: np.ndarray, to_uint16: bool = True) -> np.ndarray:
    """
    Converts the RGB stitched output into the FINAL BGR-ordered deliverable.

    I-3 (BLOCKING for deliverable): the submission TIFF MUST be BGR band order.
    stitch_inference() returns RGB (convenient for display); this helper performs
    the mandatory RGB->BGR conversion and is the only sanctioned path for writing
    the submission image.

    Args:
        rgb_image : (H, W, 3) uint8 RGB array as returned by stitch_inference().
        to_uint16 : If True (default), scale the 8-bit [0, 255] values up to the
                    project's uint16 [0, 65535] range (the documented imagery
                    range). If False, keep uint8.

    Returns:
        bgr_image : (H, W, 3) array in BGR band order. dtype is uint16 when
                    to_uint16 is True, else uint8.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(
            f"Expected (H, W, 3) RGB array for the deliverable, got shape {rgb_image.shape}"
        )

    # RGB -> BGR (mandatory eval band order). Use cv2 for clarity of intent.
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if to_uint16:
        # uint8 [0,255] -> uint16 [0,65535] using the documented /65535 contract
        # in reverse: value_16 = value_8 / 255 * 65535. NEVER a /255-only scale.
        bgr_image = (bgr_image.astype(np.float32) / 255.0 * 65535.0)
        bgr_image = np.clip(bgr_image, 0, 65535).astype(np.uint16)

    return bgr_image  # (H, W, 3) BGR — write THIS as the submission TIFF


def _accumulate_patch(patch, model, output_acc, weight_acc, gauss, gauss_3ch,
                      y, x, patch_size, device, g_min, g_max):
    """Internal helper: runs one patch through the model and accumulates."""
    if patch.max() - patch.min() < 1e-6:
        return
    patch_norm = 2.0 * (patch - g_min) / (g_max - g_min) - 1.0
    patch_norm = np.clip(patch_norm, -1.0, 1.0)
    tensor = torch.from_numpy(patch_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
    pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_np = np.clip((pred_np + 1.0) / 2.0 * 255.0, 0, 255)
    output_acc[y:y + patch_size, x:x + patch_size] += pred_np * gauss_3ch
    weight_acc[y:y + patch_size, x:x + patch_size] += gauss


# ── Optional: Global Histogram Matching (Remedy 3) ────────────────────────────

def match_histograms_global(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Matches the histogram of 'source' (your stitched output) to 'reference'
    (a known good RGB image of similar land cover).

    Corrects residual global tone drift across the full stitched image
    without blurring any edges.

    Args:
        source    : (H, W, 3) uint8 — your stitched model output
        reference : (H, W, 3) uint8 — a real RGB image used as color reference

    Returns:
        matched   : (H, W, 3) uint8 — color-corrected output
    """
    matched = np.zeros_like(source)
    for ch in range(3):
        src_ch = source[:, :, ch]
        ref_ch = reference[:, :, ch]

        # Build cumulative distribution functions
        src_hist, _ = np.histogram(src_ch.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_ch.flatten(), 256, [0, 256])

        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()

        # Normalize CDFs
        src_cdf_norm = src_cdf / src_cdf[-1]
        ref_cdf_norm = ref_cdf / ref_cdf[-1]

        # Build lookup table: for each src intensity, find matching ref intensity
        lut = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and ref_cdf_norm[ref_idx] < src_cdf_norm[src_idx]:
                ref_idx += 1
            lut[src_idx] = ref_idx

        matched[:, :, ch] = lut[src_ch]

    return matched
