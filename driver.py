import os
import shutil
import argparse
import subprocess
import numpy as np
import tifffile
from utils.logging_utils import setup_logging
from utils.file_utils import find_file

# 30m → 100m downscale factor (exact: 100/30 = 3.3333…).
# Using the rounded string '3.33' produced rgb/qa grids 2-3 px larger than
# the native-100m tir_100m grid, causing ~250m TIR/RGB co-registration error.
SCALE_30M_TO_100M = str(100 / 30)


def _crop_file_to(path, target_h, target_w, logger):
    """Crop a (multi-band) image file in-place to target_h × target_w."""
    img = tifffile.imread(path)
    h, w = img.shape[-2:]
    if (h, w) == (target_h, target_w):
        return
    if h < target_h or w < target_w:
        logger.warning(
            f"{os.path.basename(path)} ({h}x{w}) smaller than target "
            f"({target_h}x{target_w}); cannot crop up.")
        return
    cropped = img[..., :target_h, :target_w]
    tifffile.imwrite(path, cropped.astype(img.dtype))
    logger.info(f"Cropped {os.path.basename(path)} {h}x{w} -> {target_h}x{target_w}")


def align_100m_grids(tir_100m_path, rgb_100m_path, qa_100m_path, tir_200m_path, logger):
    """Align all 100m products to the minimum common H x W grid.

    The native B10 (tir_100m) and the downscaled rgb/qa (30m / 3.333) can
    differ by +/-1 px due to independent int(round()) rounding. This function
    crops ALL products to min(H), min(W) so create_patches.py's co-registration
    guard passes. tir_200m is also cropped to min_h//2, min_w//2 to maintain
    the exact 2x relationship."""
    paths_100m = [tir_100m_path, rgb_100m_path]
    if qa_100m_path:
        paths_100m.append(qa_100m_path)

    dims = []
    for p in paths_100m:
        img = tifffile.imread(p)
        dims.append(img.shape[-2:])

    min_h = min(d[0] for d in dims)
    min_w = min(d[1] for d in dims)

    if all(d == (min_h, min_w) for d in dims):
        return  # already aligned

    logger.info(f"Aligning 100m grids to common {min_h}x{min_w} "
                f"(dims were: {dims})")

    for p in paths_100m:
        _crop_file_to(p, min_h, min_w, logger)

    # Keep tir_200m at exactly half the aligned 100m grid
    _crop_file_to(tir_200m_path, min_h // 2, min_w // 2, logger)

def run_script(script_name, logger, *args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, 'scripts')
    script_path = os.path.join(scripts_dir, script_name)
    command = ['python', script_path] + list(args)
    logger.info(f"Running: {' '.join(command)}")
    try:
        # Add the project root to PYTHONPATH so scripts can import from utils
        env = os.environ.copy()
        env['PYTHONPATH'] = base_dir + os.pathsep + env.get('PYTHONPATH', '')
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
        if result.stdout:
            logger.info(f"STDOUT from {script_name}:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR from {script_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise e

def main():
    parser = argparse.ArgumentParser(description='IR-Colorization Dataset Generation Baseline')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    
    output_downscale_dir = os.path.join(output_dir, 'downscaled_data')
    output_rgb_dir = os.path.join(output_dir, 'rgb_images')
    output_patches_dir = os.path.join(output_dir, 'patches')

    for d in [output_downscale_dir, output_rgb_dir, output_patches_dir]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(log_dir=output_dir)

    if not os.path.isdir(input_root):
        logger.error(f"Input root directory {input_root} not found.")
        exit(1)

    product_folders = [e for e in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, e))]

    for product_id in product_folders:
        input_dir = os.path.join(input_root, product_id)
        logger.info(f"Processing product: {product_id}")

        band2_path  = find_file(input_dir, '_B2')
        band3_path  = find_file(input_dir, '_B3')
        band4_path  = find_file(input_dir, '_B4')
        band10_path = find_file(input_dir, '_B10')
        qa_path     = find_file(input_dir, '_QA_PIXEL')

        if not all([band2_path, band3_path, band4_path, band10_path]):
            logger.warning(f"Skipping {product_id}: Missing required bands.")
            continue

        file_prefix = product_id

        try:
            # 1. Merge RGB (30m) from SR_B4 + SR_B3 + SR_B2
            rgb_output_path = os.path.join(output_rgb_dir, f'{file_prefix}_rgb_30m.tif')
            run_script('merge_rgb.py', logger, band4_path, band3_path, band2_path, rgb_output_path)

            # 2. Downscale RGB 30m → 100m (exact 100/30x box average)
            downscaled_rgb_100m = os.path.join(output_downscale_dir, f'{file_prefix}_rgb_100m.tif')
            run_script('downscale.py', logger, rgb_output_path, downscaled_rgb_100m, SCALE_30M_TO_100M)

            # 3. ST_B10 was re-exported from GEE at scale=100 (native TIRS resolution).
            #    No downscaling needed — copy it directly as tir_100m.
            #    (The old pipeline downscaled 30m→100m at 3.33x, which only worked on
            #    bicubically upsampled data from GEE and produced fake SR ground truth.)
            downscaled_tir_100m = os.path.join(output_downscale_dir, f'{file_prefix}_tir_100m.tif')
            shutil.copy2(band10_path, downscaled_tir_100m)
            logger.info(f"Copied native 100m B10 → {downscaled_tir_100m}")

            # 4. Downscale TIR 100m → 200m (2.0x box average) — genuine 2x SR pair
            downscaled_tir_200m = os.path.join(output_downscale_dir, f'{file_prefix}_tir_200m.tif')
            run_script('downscale.py', logger, downscaled_tir_100m, downscaled_tir_200m, '2.0')

            # 5. Downscale QA_PIXEL 30m → 100m using nearest-neighbor (categorical data)
            downscaled_qa_100m = None
            if qa_path:
                downscaled_qa_100m = os.path.join(output_downscale_dir, f'{file_prefix}_qa_100m.tif')
                run_script('downscale.py', logger, qa_path, downscaled_qa_100m, SCALE_30M_TO_100M, '--method', 'nearest')
            else:
                logger.warning(f"{product_id}: QA_PIXEL not found — cloud masking disabled for this scene")

            # 6. Force all 100m products onto the same H×W grid.
            #    Native B10 and downscaled rgb/qa can differ by ±1 px due to
            #    independent int(round(dim / 3.3333)) rounding. Crop ALL to
            #    min(H), min(W) so the patch loop's co-registration guard passes.
            align_100m_grids(downscaled_tir_100m, downscaled_rgb_100m,
                             downscaled_qa_100m, downscaled_tir_200m, logger)

            logger.info(f"Successfully generated intermediate files for {product_id}")

        except Exception as e:
            logger.error(f"Error processing {product_id}: {e}")

    # 5. Create Coregistered Patches (Run ONCE after all scenes are downscaled)
    try:
        run_script('create_patches.py', logger, '--input_dir', output_downscale_dir, '--output_dir', output_patches_dir)
    except Exception as e:
        logger.error(f"Error running patch extraction: {e}")

    logger.info("Dataset generation finished. Samples available in output/patches")

if __name__ == '__main__':
    main()