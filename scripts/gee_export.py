import ee
import time
import os
import json
import argparse


# ── Region definitions ──────────────────────────────────────────────────────
# Focused 2.5° × 2.5° bounding boxes (~275 km × 275 km) centered on
# representative terrain for each land-cover class across India.
# Sorted by: sort('CLOUD_COVER') → least-cloudy scenes exported first.

EXPORT_REGIONS = {
    # Terrain datasets you want to train with
    # Format: 'terrain_name': {'bbox': [lon_min, lat_min, lon_max, lat_max], 'scenes': n}
    #eg: 'thar_desert':       {'bbox': [69.5, 25.5, 72.0, 28.0], 'scenes': 8},
}

# ── Band configuration ──────────────────────────────────────────────────────
# Maps GEE L2 band names → pipeline-friendly names (driver.py / organize_scenes.py)
BAND_MAP = {
    #bands you want
    #format: 'GEE_BAND_NAME': 'DRIVER_BAND_NAME'
    #eg:'ST_B10':   'B10',
}

ALL_BAND_NAMES = list(BAND_MAP.values())   # ['B10', 'B2', 'B3', 'B4', 'QA_PIXEL']
DATA_BANDS     = ['B2', 'B3', 'B4', 'B10']
DRIVE_FOLDER   = 'ISRO_PS10_L9_v2'

# Skip submitting a scene whose valid (non-masked) pixel fraction over the AOI
# falls below this — rectangular AOIs that extend past the Landsat swath
# otherwise export ~98% 0-fill black border.
MIN_VALID_FRACTION = 0.30

# Minimum fraction of the AOI bbox a single Landsat acquisition's footprint must
# cover to be eligible. The 2.5° (~275 km) bbox is LARGER than one ~185 km WRS-2
# footprint, so a single scene covers AT MOST ~48% of the box — a 0.60 cutoff is
# physically unreachable and rejects every scene (the 0/83 dry-run bug). 0.30
# keeps genuine well-covering scenes while discarding corner-clip / swath-edge
# images (the ~98%-nodata bug). Override per-run with --min-coverage; the dry-run
# prints the best available coverage per region so this is tuned from data.
MIN_COVERAGE = 0.30

# Upper bound on images pulled to the client per region for the path/row de-dup
# loop. The collection is already sorted by CLOUD_COVER and coverage-filtered
# server-side, so the least-cloudy qualifying images are at the head of the list;
# 200 is far more than any single bbox yields in the 2022–2026 window.
CLIENT_LIST_LIMIT = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export Landsat 9 L2 training scenes from GEE to Google Drive.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python gee_export.py --project <your-google-earth-engine-project-id>\n'
               '  python gee_export.py --project <your-google-earth-engine-project-id> --bands QA_PIXEL\n'
    )
    parser.add_argument('--project', required=True,
                        help='Google Earth Engine project ID )')
    parser.add_argument('--bands', default=','.join(ALL_BAND_NAMES),
                        help=f'Comma-separated bands to export (default: {",".join(ALL_BAND_NAMES)})')
    parser.add_argument('--folder', default=DRIVE_FOLDER,
                        help=f'Google Drive folder name (default: {DRIVE_FOLDER})')
    parser.add_argument('--regions', default=None,
                        help='Comma-separated terrain names to export (default: all). '
                             'e.g. thar_desert,himalayan_snow . Validated against '
                             'EXPORT_REGIONS keys; unknown names are rejected.')
    parser.add_argument('--min-coverage', type=float, default=MIN_COVERAGE,
                        help=f'Minimum AOI-coverage fraction for a single acquisition '
                             f'to qualify (default: {MIN_COVERAGE}). A 2.5° bbox exceeds '
                             f'one ~185 km Landsat footprint, so the max achievable is '
                             f'~0.48 — lower this if a region reports 0 qualifying.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be exported without submitting tasks')
    parser.add_argument('--write-manifest', default=None, metavar='PATH',
                        help='Write a JSON manifest of CHOSEN scenes (one record per '
                             'scene: scene_label, terrain, path, row, date, asset_id, '
                             'coverage, filled) to PATH. Works in BOTH --dry-run and '
                             'real runs (selection is identical), so the manifest can '
                             'be regenerated with --dry-run --write-manifest <path> '
                             'without re-submitting export tasks. Parent dirs created.')
    return parser.parse_args()


def build_collection(requested_bands):
    """Build the base Landsat 9 L2 collection with requested bands."""
    # Map driver-style names back to GEE names for .select()
    driver_to_gee = {v: k for k, v in BAND_MAP.items()}
    gee_bands = [driver_to_gee[b] for b in requested_bands]

    return (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterDate('2022-10-01', '2026-04-30')
        .filter(ee.Filter.lt('CLOUD_COVER', 10))
        .select(gee_bands)
    )


def estimate_valid_fraction(image, geometry, mask_band):
    try:
        mask = image.select([mask_band]).mask().rename('valid')
        result = mask.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = geometry,
            scale     = 100,     # coarse scale keeps this lightweight
            maxPixels = 1e9,
            bestEffort= True,
        ).getInfo()
        if result is None or result.get('valid') is None:
            return None
        return float(result['valid'])
    except Exception as e:
        print(f'    [warn] valid-fraction estimate failed: {e}')
        return None


def select_coherent_scenes(collection, geometry, n_scenes, region_name, min_coverage):
    aoi_area = geometry.area(maxError=1)

    def tag_coverage(img):
        # Fraction of the AOI bbox covered by this acquisition's footprint.
        # intersection(...).area / aoi_area ∈ [0, 1]. A single ~185 km Landsat
        # footprint covers AT MOST ~0.48 of a 2.5° box — never 1.0.
        inter = img.geometry().intersection(geometry, ee.ErrorMargin(1)).area(maxError=1)
        return img.set('aoi_coverage', inter.divide(aoi_area))

    # Tag every intersecting image with its AOI coverage, then cloud-sort. We do
    # NOT coverage-filter server-side: pulling the full coverage distribution to
    # the client lets us REPORT what is actually available (and tune
    # --min-coverage from data) instead of silently returning nothing.
    tagged = (
        collection
        .filterBounds(geometry)
        .map(tag_coverage)
        .sort('CLOUD_COVER')
    )

    try:
        feats = tagged.toList(CLIENT_LIST_LIMIT).getInfo()
    except Exception as e:
        print(f'  [warn] {region_name}: scene query failed ({e}); skipping region.')
        return []

    if not feats:
        print(f'  [info] {region_name}: no Landsat acquisitions intersect the AOI '
              f'(CLOUD_COVER < 10, 2022-2026).')
        return []

    # Build the client-side candidate list (already least-cloudy-first).
    candidates = []
    for feat in feats:
        props = feat.get('properties', {}) or {}
        asset_id = feat.get('id')
        cov = props.get('aoi_coverage')
        if asset_id is None or cov is None:
            continue
        candidates.append({
            'path': props.get('WRS_PATH'),
            'row': props.get('WRS_ROW'),
            'cloud': props.get('CLOUD_COVER'),
            'coverage': cov,
            'date': asset_id.split('/')[-1],   # e.g. LC09_147039_20231012
            'id': asset_id,
        })

    # Diagnostic: surface the best-covering acquisitions so the threshold can be
    # tuned from data rather than guessed.
    by_cov = sorted(candidates, key=lambda m: m['coverage'], reverse=True)
    top_summary = ', '.join(f"{m['coverage']:.2f}" for m in by_cov[:5])
    best_cov = by_cov[0]['coverage'] if by_cov else 0.0

    # Keep only images covering >= min_coverage that carry a usable WRS-2 tile id.
    qualifying = [m for m in candidates
                  if m['coverage'] >= min_coverage
                  and m['path'] is not None and m['row'] is not None]

    if not qualifying:
        print(f"  [WARNING] {region_name}: 0 of {len(candidates)} intersecting "
              f"acquisitions reach {min_coverage:.0%} AOI coverage "
              f"(best {best_cov:.2f}; top-5: {top_summary}). "
              f"Lower --min-coverage or shrink this region's bbox.")
        return []

    # Path/row de-dup over the cloud-sorted qualifying list.
    seen_tiles = set()
    primary = []     # one image per distinct (path,row)
    secondary = []   # fills: extra qualifying images from already-seen tiles
    for meta in qualifying:          # qualifying preserves cloud-sorted order
        tile = (meta['path'], meta['row'])
        if tile not in seen_tiles:
            seen_tiles.add(tile)
            primary.append(meta)
        else:
            secondary.append(meta)

    # Prefer distinct path/rows; fill remainder with same-tile other dates.
    chosen_meta = []
    for meta in primary:
        if len(chosen_meta) >= n_scenes:
            break
        meta['filled'] = False
        chosen_meta.append(meta)

    n_filled = 0
    if len(chosen_meta) < n_scenes:
        for meta in secondary:          # already least-cloudy-first
            if len(chosen_meta) >= n_scenes:
                break
            meta['filled'] = True
            chosen_meta.append(meta)
            n_filled += 1

    if n_filled:
        print(f'  [info] {region_name}: {len(primary)} distinct WRS-2 tile(s) '
              f'qualified; filled {n_filled} scene(s) from already-seen tiles '
              f'(other acquisition dates) to reach the target.')

    # Reconstruct each chosen image server-side, exact, by its asset id.
    chosen = [(ee.Image(m['id']), m) for m in chosen_meta]
    return chosen


def export_scenes(collection, requested_bands, drive_folder, dry_run=False,
                  regions=None, min_coverage=MIN_COVERAGE, manifest=None):
    """Submit per-band export tasks to Google Drive.

    If `manifest` is a list, append one provenance record per CHOSEN scene
    (scene_label, terrain, path, row, date, asset_id, coverage, filled). The
    accumulation happens at selection time and is independent of --dry-run, so a
    dry run produces the same manifest as a real run."""
    # Reverse map for renaming
    driver_to_gee = {v: k for k, v in BAND_MAP.items()}
    gee_bands = [driver_to_gee[b] for b in requested_bands]

    # Restrict to the requested subset of regions (default: all). `regions` is a
    # validated list of EXPORT_REGIONS keys (validated in main()); iterate in
    # EXPORT_REGIONS definition order for a stable, reproducible run.
    if regions is None:
        regions = list(EXPORT_REGIONS.keys())
    selected_regions = [r for r in EXPORT_REGIONS if r in regions]

    submitted = 0
    skipped_lowvalid = 0
    # NOTE: total_scenes is the ALLOCATED target for the selected subset. Fewer
    # may actually be exported if a region has too few qualifying acquisitions —
    # that shortfall is reported per-region as a WARNING and in the summary.
    total_scenes = sum(EXPORT_REGIONS[r]['scenes'] for r in selected_regions)
    total_tasks = total_scenes * len(requested_bands)
    missing_scenes = 0   # allocated-but-unavailable scenes across the subset

    # Band used to estimate valid-pixel fraction: prefer a data band, else
    # fall back to whatever was requested (e.g. a QA_PIXEL-only supplement run).
    mask_band = next((b for b in requested_bands if b in DATA_BANDS), requested_bands[0])

    print(f"{'[DRY RUN] ' if dry_run else ''}Exporting up to {total_scenes} scenes × "
          f"{len(requested_bands)} bands = {total_tasks} tasks\n")
    print(f"  Drive folder: {drive_folder}/")
    print(f"  Regions: {selected_regions}")
    print(f"  Bands: {requested_bands}\n")
    print(f"  {'Scene':<30} {'Band':<12} {'Status'}")
    print(f"  {'-'*54}")

    for region_name in selected_regions:
        config = EXPORT_REGIONS[region_name]
        geometry = ee.Geometry.Rectangle(config['bbox'])
        n_scenes = config['scenes']

        # Path/row-grouped, coverage-filtered, single coherent acquisitions.
        chosen = select_coherent_scenes(collection, geometry, n_scenes, region_name,
                                        min_coverage)

        if len(chosen) < n_scenes:
            shortfall = n_scenes - len(chosen)
            missing_scenes += shortfall
            print(f'  [WARNING] {region_name}: only {len(chosen)} of {n_scenes} '
                  f'allocated scenes qualified (>= {min_coverage:.0%} AOI coverage, '
                  f'CLOUD_COVER < 10) — exporting what is available, not '
                  f'fabricating {shortfall} scene(s).')

        for i, (image, meta) in enumerate(chosen):
            scene_label = f'{region_name}_{i + 1:02d}'
            # One-line provenance for the chosen acquisition (path/row de-dup).
            cov = meta.get('coverage')
            cld = meta.get('cloud')
            cov_s = f'{cov:.2f}' if isinstance(cov, (int, float)) else 'n/a'
            cld_s = f'{cld:.1f}' if isinstance(cld, (int, float)) else 'n/a'
            print(f"  [pick] {scene_label}: {meta['date']} p{meta['path']}/r{meta['row']} "
                  f"cloud={cld_s}% coverage={cov_s}"
                  f"{' (fill: same tile, other date)' if meta.get('filled') else ''}")

            # Persist the scene → WRS-2 tile mapping. organize_scenes.py renames
            # files to <scene>_<band>.TIF and DROPS the path/row/asset-id, so this
            # is the only point the (path,row) tile a scene belongs to is known.
            # Recorded for every chosen scene, regardless of dry-run, because the
            # downstream tile-grouped split needs it.
            if manifest is not None:
                manifest.append({
                    'scene_label': scene_label,
                    'terrain':     region_name,
                    'path':        meta.get('path'),
                    'row':         meta.get('row'),
                    'date':        meta.get('date'),
                    'asset_id':    meta.get('id'),
                    'coverage':    meta.get('coverage'),
                    'filled':      bool(meta.get('filled', False)),
                })

            # Rename GEE bands → driver.py names
            image_renamed = image.select(gee_bands, requested_bands)

            # Clip to the AOI intersected with the scene's own footprint so
            # corners outside the Landsat swath are dropped, not exported as
            # 0-fill black border (the ~98%-nodata bug).
            export_region = geometry.intersection(image.geometry(), ee.ErrorMargin(1))
            image_renamed = image_renamed.clip(export_region)

            # Pre-export valid-fraction estimate over the AOI (server-side).
            valid_frac = None if dry_run else estimate_valid_fraction(
                image_renamed, geometry, mask_band)
            skip_scene = False
            if valid_frac is not None:
                if valid_frac < MIN_VALID_FRACTION:
                    print(f'  [WARNING] {scene_label}: valid fraction '
                          f'{valid_frac:.3f} < {MIN_VALID_FRACTION} over AOI '
                          f'{config["bbox"]} — skipping (likely off-swath).')
                    skip_scene = True
                    skipped_lowvalid += len(requested_bands)
                else:
                    print(f'  [info] {scene_label}: AOI valid fraction '
                          f'{valid_frac:.3f}')
            elif not dry_run:
                # Diagnostic fallback so a future run is still debuggable.
                print(f'  [info] {scene_label}: AOI {config["bbox"]} — valid '
                      f'fraction unavailable, proceeding.')

            for band in requested_bands:
                task_desc = f'{scene_label}_{band}'

                if dry_run:
                    print(f'  {scene_label:<30} {band:<12} [would submit]')
                    submitted += 1
                    continue
                if skip_scene:
                    print(f'  {scene_label:<30} {band:<12} skipped (low valid)')
                    continue

                # B10 (ST_B10) native resolution is 100m; exporting at 30m
                # causes GEE to bicubically upsample it, creating fake spatial
                # detail that destroys the SR training signal.
                export_scale = 100 if band == 'B10' else 30
                task = ee.batch.Export.image.toDrive(
                    image          = image_renamed.select([band]),
                    description    = task_desc,
                    folder         = drive_folder,
                    fileNamePrefix = task_desc,
                    scale          = export_scale,
                    region         = geometry,
                    fileFormat     = 'GeoTIFF',
                    maxPixels      = 1e9,
                )
                task.start()
                print(f'  {scene_label:<30} {band:<12} submitted')
                time.sleep(0.3)

                submitted += 1

    return submitted, total_tasks, skipped_lowvalid, missing_scenes


def main():
    args = parse_args()

    # Parse requested bands
    requested_bands = [b.strip() for b in args.bands.split(',')]
    for b in requested_bands:
        if b not in ALL_BAND_NAMES:
            print(f"ERROR: Unknown band '{b}'. Valid bands: {ALL_BAND_NAMES}")
            return

    # Parse + validate requested regions (default: all). Reject unknown names so
    # a typo never silently exports nothing / the wrong subset.
    regions = None
    if args.regions:
        regions = [r.strip() for r in args.regions.split(',') if r.strip()]
        unknown = [r for r in regions if r not in EXPORT_REGIONS]
        if unknown:
            print(f"ERROR: Unknown region(s) {unknown}. "
                  f"Valid regions: {list(EXPORT_REGIONS.keys())}")
            return

    # Initialize GEE
    ee.Initialize(project=args.project)
    print(f"GEE initialized: {args.project}\n")

    # Build collection and export
    collection = build_collection(requested_bands)
    manifest = [] if args.write_manifest else None
    submitted, total, skipped_lowvalid, missing_scenes = export_scenes(
        collection, requested_bands, args.folder, args.dry_run, regions=regions,
        min_coverage=args.min_coverage, manifest=manifest)

    # Persist the scene → WRS-2 tile manifest (used by the tile-grouped split).
    if args.write_manifest is not None:
        out_dir = os.path.dirname(os.path.abspath(args.write_manifest))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.write_manifest, 'w') as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nWrote {len(manifest)} scene record(s) to manifest: "
              f"{args.write_manifest}")

    # Summary
    print(f"\n{'='*58}")
    if args.dry_run:
        print(f"[DRY RUN] Would submit {submitted} export tasks "
              f"(allocated target: {total})")
    else:
        print(f"Submitted {submitted} export tasks to Drive folder '{args.folder}'")
        if skipped_lowvalid:
            print(f"Skipped {skipped_lowvalid} tasks "
                  f"(scenes with valid fraction < {MIN_VALID_FRACTION})")
    if missing_scenes:
        print(f"WARNING: {missing_scenes} allocated scene(s) had no qualifying "
              f"acquisition (>= {args.min_coverage:.0%} AOI coverage) and were not "
              f"fabricated — exported fewer than the allocation for those regions.")
    print(f"Monitor progress: https://code.earthengine.google.com/tasks")
    print(f"\nAfter all tasks complete, run:")
    print(f"  python scripts/organize_scenes.py --src {args.folder} --dst ./input")


if __name__ == '__main__':
    main()
