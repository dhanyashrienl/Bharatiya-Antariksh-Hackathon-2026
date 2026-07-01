import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def percentile_stretch(image, low=2, high=98):
    """
    Stretches the intensity of an image based on percentiles to remove outliers.
    """

    low_val = np.percentile(image, low)
    high_val = np.percentile(image, high)

    # Flat-image guard: a constant / nodata-filled band has high_val == low_val
    # (or worse). The original code divided by (high-low+1e-5), silently mapping
    # everything to near-zero. Detect this explicitly and return a zeros array of
    # the same shape so the caller knows the band carried no usable dynamic range.
    if high_val <= low_val:
        logger.warning(
            "percentile_stretch: flat band (p%s=%s >= p%s=%s); returning zeros.",
            high, high_val, low, low_val,
        )
        return np.zeros(image.shape, dtype=np.uint8)

    stretched = np.clip(image, low_val, high_val)
    stretched = (stretched - low_val) * 255.0 / (high_val - low_val + 1e-5)

    return stretched.astype(np.uint8)
