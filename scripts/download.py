import ee
import geemap
import os
import argparse
import logging

logger = logging.getLogger(__name__)

def download_landsat_data(product_id, bands, start_date, end_date, output_path,
                          ee_project_id=None, wrs_path=None, wrs_row=None, bbox=None):
    if ee_project_id:
        ee.Initialize(project=ee_project_id)
    else:
        ee.Initialize()

    collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterDate(start_date, end_date)

    # Location selection is no longer hardcoded. Prefer an explicit bbox if
    # given; otherwise fall back to WRS path/row when both are supplied.
    if bbox is not None:
        region = ee.Geometry.Rectangle(bbox)
        collection = collection.filterBounds(region)
    else:
        if wrs_path is not None:
            collection = collection.filterMetadata('WRS_PATH', 'equals', wrs_path)
        if wrs_row is not None:
            collection = collection.filterMetadata('WRS_ROW', 'equals', wrs_row)
        if wrs_path is None and wrs_row is None:
            logger.warning('No location filter given (--bbox or --wrs_path/--wrs_row); '
                           'collection is unfiltered by location and .first() is non-deterministic.')

    collection = collection.select(bands)

    image = collection.first()

    if image:
        logger.info(f'Attempting to download image for product ID: {product_id} with bands {bands} to {output_path}')
        os.makedirs(output_path, exist_ok=True)
        filename_prefix = f'landsat9_{product_id}'

        export_region = ee.Geometry.Rectangle(bbox).bounds() if bbox is not None else image.geometry().bounds()
        try:
            geemap.ee_export_image(image, output_path, scale=30, region=export_region, file_per_band=True, filename_prefix=filename_prefix)
            logger.info(f'Individual band images successfully downloaded to {output_path}')
        except Exception as e:
            logger.error(f"Error during direct download using geemap: {e}")
            logger.error("Please ensure 'geemap' is installed, Earth Engine is authenticated, and check your quota or image region bounds.")
    else:
        logger.warning(f'No image found for product ID: {product_id} in the specified date range/location.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Landsat 9 data using Earth Engine and geemap.')
    parser.add_argument('product_id', type=str, help='Unique ID for the product.')
    parser.add_argument('bands', type=str, help='Comma-separated list of bands to download (e.g., SR_B2,SR_B3,SR_B4,ST_B10,ST_B11).')
    parser.add_argument('start_date', type=str, help='Start date for filtering (YYYY-MM-DD).')
    parser.add_argument('end_date', type=str, help='End date for filtering (YYYY-MM-DD).')
    parser.add_argument('output_path', type=str, help='Local path to save downloaded bands.')
    parser.add_argument('--ee_project_id', type=str, default=None, help='Google Earth Engine project ID.')
    parser.add_argument('--wrs_path', type=int, default=None,
                        help='WRS-2 path to select the scene location (no default).')
    parser.add_argument('--wrs_row', type=int, default=None,
                        help='WRS-2 row to select the scene location (no default).')
    parser.add_argument('--bbox', type=float, nargs=4, default=None,
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                        help='Bounding box (lon/lat) to select the scene location; '
                             'takes precedence over --wrs_path/--wrs_row.')

    args = parser.parse_args()

    bands_list = args.bands.split(',')
    logger.info("Ensure 'geemap' is installed (`pip install geemap`) and Earth Engine is authenticated (`ee.Authenticate()`, `ee.Initialize()`).")
    download_landsat_data(args.product_id, bands_list, args.start_date, args.end_date, args.output_path,
                          args.ee_project_id, wrs_path=args.wrs_path, wrs_row=args.wrs_row, bbox=args.bbox)
