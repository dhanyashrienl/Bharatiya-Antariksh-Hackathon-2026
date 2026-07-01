import numpy as np
import tifffile
import os
import argparse
import cv2
import logging

logger = logging.getLogger(__name__)

from utils.file_utils import validate_extension

_METHOD_MAP = {
    'area':    cv2.INTER_AREA,     # box average — correct for continuous data (TIR, RGB)
    'nearest': cv2.INTER_NEAREST,  # nearest-neighbor — required for categorical bands (QA_PIXEL)
}

def downscale_band(image, factor, interp):
    h, w = image.shape
    new_h = int(round(h / factor))
    new_w = int(round(w / factor))
    return cv2.resize(image, (new_w, new_h), interpolation=interp)

def downscale_image(input_filepath, output_filepath, scale_factor, method='area'):
    validate_extension(input_filepath)
    validate_extension(output_filepath)
    os.makedirs(os.path.dirname(output_filepath) or '.', exist_ok=True)

    interp = _METHOD_MAP.get(method, cv2.INTER_AREA)
    image_data = tifffile.imread(input_filepath)
    if image_data.ndim == 2:
        image_data = image_data[np.newaxis, ...]

    downscaled_bands = []
    for band in image_data:
        downscaled_bands.append(downscale_band(band, scale_factor, interp))

    downscaled_data = np.stack(downscaled_bands, axis=0)
    tifffile.imwrite(output_filepath, downscaled_data.astype(image_data.dtype))
    logger.info(f'Downscaled {input_filepath} by {scale_factor}x (method={method}) → {output_filepath}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downscale a TIFF image.')
    parser.add_argument('input_filepath', type=str, help='Path to the input TIFF file.')
    parser.add_argument('output_filepath', type=str, help='Path to save the downscaled TIFF file.')
    parser.add_argument('scale_factor', type=float, help='Factor by which to downscale (e.g., 2.0, 3.33).')
    parser.add_argument('--method', choices=['area', 'nearest'], default='area',
                        help='Interpolation method: area (box avg, for continuous data) or nearest (for QA_PIXEL).')

    args = parser.parse_args()

    try:
        downscale_image(args.input_filepath, args.output_filepath, args.scale_factor, args.method)
    except Exception as e:
        logger.error(f"Error downscaling image: {e}")
        exit(1)
