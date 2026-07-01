"""
create_patches.py — Co-registered Patch Extraction
====================================================
Extracts co-registered patch pairs from downscaled Landsat 9 scenes.

Correct preprocessing pipeline (re-run v2 — 2026-06-28):
  tir_200m  → input to Super-Resolution model     (1, 256, 256) uint16
  tir_100m_512 → SR ground truth                  (1, 512, 512) uint16
  rgb_100m_512 → Colorization ground truth        (3, 512, 512) uint16

Physics-informed loss support files (saved per patch):
  valid_mask.npy  — bool (True = valid, non-nodata pixel)        (1, 512, 512)
  tir_kelvin.npy  — float32, T_K = DN * 0.00341802 + 149.0      (1, 512, 512)
                    nodata pixels (DN=0) are set to 0.0 K

Per-patch metrics:
  meta.json — bicubic-baseline PSNR/SSIM, temperature stats, grid coords

Dataset-level files:
  dataset_info.json — normalization constants, scale factors, split counts

Key fixes vs v1:
  - stride 32→128  (was 87.5% overlap causing train/val data leakage)
  - QA_PIXEL cloud/shadow filter (>10% contaminated patches skipped)
  - Physics metadata baked in to avoid a 3rd re-preprocessing
  - Dataset split done by SCENE not by patch (prevents leakage)

Usage:
    python scripts/create_patches.py \\
        --input_dir  output/downscaled_data \\
        --output_dir output/patches \\
        [--stride 128] [--cloud_thresh 0.10] [--nodata_thresh 0.50]
"""

import tifffile
import numpy as np
import os
import glob
import json
import argparse
import logging
import cv2

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False

from utils.logging_utils import setup_logging
from utils.visualization import percentile_stretch
from utils.file_utils import find_file

# ── Landsat 9 C2 L2 calibration constants ────────────────────────────────────
TIR_SCALE  = 0.00341802   # ST_B10 DN → Kelvin: T_K = DN * TIR_SCALE + TIR_OFFSET
TIR_OFFSET = 149.0
SR_SCALE   = 2.75e-5      # SR_Bx DN → reflectance: R = DN * SR_SCALE + SR_OFFSET
SR_OFFSET  = -0.2

# Stefan-Boltzmann constant (W m^-2 K^-4) — for physics loss reference
SIGMA_SB   = 5.670374419e-8

# Landsat 9 Band 10 center wavelength (m) — for Planck's law
B10_LAMBDA = 10.895e-6    # 10.895 μm


def _to_kelvin(dn_arr: np.ndarray) -> np.ndarray:
    """Convert uint16 ST_B10 DN to surface temperature in Kelvin.
    Zero-DN (nodata/fill) pixels are set to 0.0 K so they are masked by valid_mask."""
    out = np.zeros(dn_arr.shape, dtype=np.float32)
    valid = dn_arr > 0
    out[valid] = dn_arr[valid].astype(np.float32) * TIR_SCALE + TIR_OFFSET
    return out


def _planck_radiance(T_K: np.ndarray) -> np.ndarray:
    """Spectral radiance B_lambda(T) at Landsat 9 B10 band center [W sr^-1 m^-2 m^-1].
    Used as a reference quantity; physics loss is typically Stefan-Boltzmann (E=sigma*T^4)
    for simplicity in training code."""
    h  = 6.62607015e-34   # Planck constant
    c  = 2.99792458e8     # speed of light
    kB = 1.380649e-23     # Boltzmann constant
    lam = B10_LAMBDA
    safe_T = np.where(T_K > 0, T_K, 1.0)   # avoid div-by-zero on nodata
    B = (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * kB * safe_T)) - 1)
    return np.where(T_K > 0, B, 0.0).astype(np.float32)


def load_rgb(directory):
    rgb_file = find_file(directory, "*100m*RGB*")
    if rgb_file:
        return tifffile.imread(rgb_file)
    b2 = find_file(directory, "*100m*B2*")
    b3 = find_file(directory, "*100m*B3*")
    b4 = find_file(directory, "*100m*B4*")
    if b2 and b3 and b4:
        return np.stack([tifffile.imread(b2), tifffile.imread(b3), tifffile.imread(b4)], axis=0)
    return None


def save_as_png(data, path):
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
    stretched = percentile_stretch(data)
    if stretched.ndim == 3:
        stretched = cv2.cvtColor(stretched, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, stretched)


def _bicubic_baseline(tir_200m_patch, tir_100m_patch):
    """Compute PSNR and SSIM of bicubic 2x upscale of 200m patch vs 100m ground truth.
    Gives the 'do nothing' baseline that the SR model must beat."""
    if not _SKIMAGE_OK:
        return None, None
    h100, w100 = tir_100m_patch.shape[-2:]
    # Squeeze to 2D for cv2 and skimage
    tir_200m_2d = tir_200m_patch.squeeze() if tir_200m_patch.ndim == 3 else tir_200m_patch
    tir_100m_2d = tir_100m_patch.squeeze() if tir_100m_patch.ndim == 3 else tir_100m_patch

    bicubic = cv2.resize(
        tir_200m_2d.astype(np.float32),
        (w100, h100),
        interpolation=cv2.INTER_CUBIC
    )
    bicubic = np.clip(bicubic, 0, 65535)
    gt = tir_100m_2d.astype(np.float32)

    psnr = peak_signal_noise_ratio(gt, bicubic, data_range=65535.0)
    ssim = structural_similarity(gt, bicubic, data_range=65535.0)
    return float(psnr), float(ssim)


def create_patches(input_root, output_root, stride=128, cloud_thresh=0.10, nodata_thresh=0.50):
    os.makedirs(output_root, exist_ok=True)
    logger = setup_logging(log_name='create_patches')

    if not os.path.exists(input_root):
        logger.error(f"Input root {input_root} does not exist.")
        return

    all_files = glob.glob(os.path.join(input_root, '*'))
    products = set()
    for f in all_files:
        filename = os.path.basename(f)
        product_id = filename
        for suffix in ['_rgb_100m.tif', '_tir_100m.tif', '_tir_200m.tif',
                        '_qa_100m.tif', '_rgb_30m.tif']:
            if filename.endswith(suffix):
                product_id = filename[:-len(suffix)]
                break
        products.add(product_id)

    logger.info(f"Found {len(products)} products in {input_root}")

    total_patches   = 0
    skipped_nodata  = 0
    skipped_cloud   = 0
    skipped_size    = 0
    scene_patch_counts = {}

    for product_id in sorted(products):
        tir_200m_path  = find_file(input_root, f'{product_id}*_tir_200m*')
        tir_100m_path  = find_file(input_root, f'{product_id}*_tir_100m*')
        rgb_100m_path  = find_file(input_root, f'{product_id}*_rgb_100m*')
        qa_100m_path   = find_file(input_root, f'{product_id}*_qa_100m*')

        if not tir_200m_path or not tir_100m_path or not rgb_100m_path:
            logger.warning(f"Skipping {product_id}: missing tir_200m / tir_100m / rgb_100m.")
            continue

        try:
            tir_200m = tifffile.imread(tir_200m_path)
            tir_100m = tifffile.imread(tir_100m_path)
            rgb_100m = tifffile.imread(rgb_100m_path)
        except Exception as e:
            logger.error(f"Error reading {product_id}: {e}")
            continue

        # Ensure CHW layout
        if tir_200m.ndim == 2: tir_200m = tir_200m[np.newaxis]
        if tir_100m.ndim == 2: tir_100m = tir_100m[np.newaxis]
        if rgb_100m.ndim == 2: rgb_100m = rgb_100m[np.newaxis]

        qa = None
        if qa_100m_path:
            qa = tifffile.imread(qa_100m_path)
            if qa.ndim == 3: qa = qa.squeeze(0)   # force 2D

        # ── Grid co-registration guard ────────────────────────────────────────
        # The patch loop assumes rgb_100m and qa_100m share tir_100m's exact
        # H×W grid (driver.py crops them to match). Fail loudly on any mismatch
        # rather than silently producing ~250m-misaligned TIR/RGB training pairs.
        tir_hw = tir_100m.shape[-2:]
        rgb_hw = rgb_100m.shape[-2:]
        if rgb_hw != tir_hw:
            logger.error(
                f"Skipping {product_id}: rgb_100m {rgb_hw} != tir_100m {tir_hw} "
                f"— 100m grids misaligned (re-run driver.py crop step).")
            continue
        if qa is not None and qa.shape[-2:] != tir_hw:
            logger.error(
                f"Skipping {product_id}: qa_100m {qa.shape[-2:]} != tir_100m {tir_hw} "
                f"— 100m grids misaligned (re-run driver.py crop step).")
            continue

        h200, w200 = tir_200m.shape[-2:]
        logger.info(f"Processing {product_id} — 200m grid {h200}×{w200}, stride={stride}")

        count = 0
        for y in range(0, h200 - 256 + 1, stride):
            for x in range(0, w200 - 256 + 1, stride):
                patch_200m_tir = tir_200m[..., y:y+256, x:x+256]

                y100, x100 = 2 * y, 2 * x
                patch_100m_tir = tir_100m[..., y100:y100+512, x100:x100+512]
                patch_100m_rgb = rgb_100m[..., y100:y100+512, x100:x100+512]

                # Shape check
                if patch_100m_tir.shape[-2:] != (512, 512) or patch_100m_rgb.shape[-2:] != (512, 512):
                    skipped_size += 1
                    continue

                # Nodata filter: skip if >nodata_thresh of 200m TIR pixels are zero
                nodata_frac = (patch_200m_tir == 0).mean()
                if nodata_frac > nodata_thresh:
                    skipped_nodata += 1
                    continue

                # Cloud/shadow filter using QA_PIXEL bit flags
                if qa is not None:
                    patch_qa = qa[y100:y100+512, x100:x100+512]
                    cloud_bit  = (patch_qa & (1 << 3)).astype(bool)
                    shadow_bit = (patch_qa & (1 << 4)).astype(bool)
                    contaminated = (cloud_bit | shadow_bit).mean()
                    if contaminated > cloud_thresh:
                        skipped_cloud += 1
                        continue

                # ── Compute physics-loss support data ─────────────────────────
                valid_mask  = (patch_100m_tir > 0).astype(np.bool_)       # (1, 512, 512)
                tir_kelvin  = _to_kelvin(patch_100m_tir)                   # (1, 512, 512) float32

                valid_K     = tir_kelvin[valid_mask]
                mean_K      = float(valid_K.mean())  if valid_K.size > 0 else 0.0
                min_K       = float(valid_K.min())   if valid_K.size > 0 else 0.0
                max_K       = float(valid_K.max())   if valid_K.size > 0 else 0.0

                # ── Bicubic baseline PSNR/SSIM ────────────────────────────────
                psnr_bicubic, ssim_bicubic = _bicubic_baseline(patch_200m_tir, patch_100m_tir)

                # ── Save files ────────────────────────────────────────────────
                sample_dir = os.path.join(output_root, product_id, f'sample_{count:05d}')
                os.makedirs(sample_dir, exist_ok=True)

                np.save(os.path.join(sample_dir, 'tir_200m.npy'),      patch_200m_tir)
                np.save(os.path.join(sample_dir, 'tir_100m_512.npy'),  patch_100m_tir)
                np.save(os.path.join(sample_dir, 'rgb_100m_512.npy'),  patch_100m_rgb)
                np.save(os.path.join(sample_dir, 'valid_mask.npy'),    valid_mask)
                np.save(os.path.join(sample_dir, 'tir_kelvin.npy'),    tir_kelvin)

                # PNG visualization (8-bit, not for training)
                for name, arr in [('tir_200m',     patch_200m_tir),
                                   ('tir_100m_512', patch_100m_tir),
                                   ('rgb_100m_512', patch_100m_rgb)]:
                    save_as_png(arr, os.path.join(sample_dir, f'{name}.png'))

                # Per-patch metadata
                meta = {
                    'product_id':            product_id,
                    'sample_id':             count,
                    'grid_y_200m':           y,
                    'grid_x_200m':           x,
                    'valid_pixel_fraction':  float(valid_mask.mean()),
                    'tir_mean_dn_200m':      float(patch_200m_tir[patch_200m_tir > 0].mean())
                                             if (patch_200m_tir > 0).any() else 0.0,
                    'tir_mean_kelvin':       mean_K,
                    'tir_min_kelvin':        min_K,
                    'tir_max_kelvin':        max_K,
                    'bicubic_baseline_psnr': psnr_bicubic,
                    'bicubic_baseline_ssim': ssim_bicubic,
                    'stefan_boltzmann_E_mean_Wm2': float(SIGMA_SB * (mean_K ** 4)) if mean_K > 0 else 0.0,
                }
                with open(os.path.join(sample_dir, 'meta.json'), 'w') as fh:
                    json.dump(meta, fh, indent=2)

                count += 1

        logger.info(f"  {product_id}: {count} patches saved.")
        if count == 0:
            # Distinguish a genuinely-tight/cloudy scene from an empty/all-nodata export.
            scene_valid_frac = float((tir_200m > 0).mean())
            # Minimum achievable per-patch nodata fraction across the slide window
            # (the best any patch could do under the nodata_thresh filter).
            min_nodata = 1.0
            for y in range(0, h200 - 256 + 1, stride):
                for x in range(0, w200 - 256 + 1, stride):
                    p = tir_200m[..., y:y+256, x:x+256]
                    if p.shape[-2:] == (256, 256):
                        min_nodata = min(min_nodata, float((p == 0).mean()))
            logger.warning(
                f"  {product_id}: 0 patches saved — scene valid fraction "
                f"{scene_valid_frac:.3f}, min achievable patch nodata fraction "
                f"{min_nodata:.3f} (nodata_thresh={nodata_thresh}, "
                f"cloud_thresh={cloud_thresh}). "
                f"Low valid fraction ⇒ tight/empty export; high ⇒ cloud-dominated.")
        scene_patch_counts[product_id] = count
        total_patches += count

    # ── Dataset-level info ────────────────────────────────────────────────────
    dataset_info = {
        'version': 2,
        'description': 'Landsat 9 C2 L2 co-registered patches — re-preprocessed 2026-06-28',
        'stride': stride,
        'patch_size_200m': 256,
        'patch_size_100m': 512,
        'total_patches': total_patches,
        'skipped': {
            'nodata': skipped_nodata,
            'cloud_shadow': skipped_cloud,
            'size': skipped_size,
        },
        'scene_patch_counts': scene_patch_counts,
        'normalization': {
            'tir_divide_by': 65535,
            'tir_to_kelvin': {'scale': TIR_SCALE, 'offset': TIR_OFFSET,
                               'formula': 'T_K = DN * 0.00341802 + 149.0'},
            'sr_reflectance': {'scale': SR_SCALE, 'offset': SR_OFFSET,
                                'formula': 'R = DN * 2.75e-5 - 0.2 (clip to [0,1])'},
            'model_input_range': '[-1, 1]  (apply: x = 2*(DN/65535) - 1)',
        },
        'physics_loss': {
            'valid_mask_file': 'valid_mask.npy',
            'kelvin_file': 'tir_kelvin.npy',
            'stefan_boltzmann_sigma': SIGMA_SB,
            'planck_lambda_m': B10_LAMBDA,
            'loss_masking': 'valid_mask = (tir_target > 0)  — zero out nodata pixels in loss',
        },
        'fid_reference_stats': 'Run scripts/compute_fid_stats.py on Kaggle (requires PyTorch)',
        'train_val_split': 'Split by SCENE (not by patch) to prevent data leakage',
        'skimage_available': _SKIMAGE_OK,
    }
    info_path = os.path.join(output_root, 'dataset_info.json')
    with open(info_path, 'w') as fh:
        json.dump(dataset_info, fh, indent=2)

    logger.info(
        f"\n{'='*60}\n"
        f"Patch extraction complete\n"
        f"  Total patches : {total_patches}\n"
        f"  Skipped nodata: {skipped_nodata}\n"
        f"  Skipped cloud : {skipped_cloud}\n"
        f"  Skipped size  : {skipped_size}\n"
        f"  Dataset info  : {info_path}\n"
        f"{'='*60}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract co-registered patch pairs.')
    parser.add_argument('--input_dir',     default='output/downscaled_data',
                        help='Directory with downscaled TIF files from driver.py.')
    parser.add_argument('--output_dir',    default='output/patches',
                        help='Output directory for patch samples.')
    parser.add_argument('--stride',        type=int, default=128,
                        help='Sliding-window stride in 200m pixels (default 128 = 50%% overlap).')
    parser.add_argument('--cloud_thresh',  type=float, default=0.10,
                        help='Skip patch if fraction of cloud+shadow pixels > this (default 0.10).')
    parser.add_argument('--nodata_thresh', type=float, default=0.50,
                        help='Skip patch if fraction of zero-DN pixels > this (default 0.50).')
    args = parser.parse_args()
    create_patches(args.input_dir, args.output_dir, args.stride, args.cloud_thresh, args.nodata_thresh)
