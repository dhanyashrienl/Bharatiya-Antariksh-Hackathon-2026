"""
organize_scenes.py
==================
Organizes flat GEE-exported files into the per-scene folder structure expected
by driver.py. Works in two modes:

FULL mode (default — first-time setup):
    Requires all REQUIRED_BANDS (B2, B3, B4, B10) per scene.
    Input (flat Drive download):
        ISRO_PS10_L9/
            thar_desert_01_B2.TIF
            thar_desert_01_B10.TIF
            ...
    Output:
        input/
            thar_desert_01/
                thar_desert_01_B2.TIF
                thar_desert_01_B10.TIF

UPDATE mode (--update — partial re-export):
    Copies only the bands present in src_dir, overwriting existing files in dst.
    Use this after re-exporting only B10 (scale=100 fix) without needing B2/B3/B4.

    python organize_scenes.py --src <new_b10_folder> --dst ./input --update

    WARNING: If both a .TIF and .tif of the same scene+band exist in dst,
    delete the old one manually before running --update to avoid ambiguity.

Usage:
    python organize_scenes.py --src ISRO_PS10_L9 --dst ./input          # full
    python organize_scenes.py --src new_b10_only --dst ./input --update  # partial
"""

import shutil
import argparse
from pathlib import Path
from collections import defaultdict

REQUIRED_BANDS = {'B2', 'B3', 'B4', 'B10'}   # driver.py needs all four
ALL_BANDS      = REQUIRED_BANDS | OPTIONAL_BANDS


def organize(src_dir: Path, dst_dir: Path, dry_run: bool = False,
             update_mode: bool = False) -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    # Single case-insensitive match. On NTFS, glob('*.TIF') + glob('*.tif')
    # returns the same files twice, double-counting scenes (I-15).
    tif_files = sorted(p for p in src_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'})
    if not tif_files:
        print(f"No .TIF/.tif files found in {src_dir}")
        return

    scenes = defaultdict(dict)
    skipped = []

    for f in tif_files:
        stem = f.stem
        if stem.endswith('_QA_PIXEL'):
            scene_label = stem[: -len('_QA_PIXEL')]
            band = 'QA_PIXEL'
        else:
            parts = stem.rsplit('_', 1)
            if len(parts) != 2 or parts[1] not in ALL_BANDS:
                skipped.append(f.name)
                continue
            scene_label, band = parts[0], parts[1]
        scenes[scene_label][band] = f

    if skipped:
        print(f"Skipped {len(skipped)} unrecognized files: {skipped[:5]}...")

    mode_label = 'UPDATE' if update_mode else 'FULL'
    print(f"\n[{mode_label} MODE] Found {len(scenes)} scenes, {len(tif_files)} TIF files\n")

    complete = 0
    incomplete = []

    for scene_label, band_files in sorted(scenes.items()):
        scene_dst = dst_dir / scene_label

        if not update_mode:
            # Full mode: require all four data bands
            missing_required = REQUIRED_BANDS - set(band_files.keys())
            missing_optional = OPTIONAL_BANDS - set(band_files.keys())
            if missing_required:
                incomplete.append((scene_label, missing_required))
                print(f'  [SKIP]  {scene_label:<30} missing required: {missing_required}')
                continue
            if missing_optional:
                print(f'  [WARN]  {scene_label:<30} missing optional: {missing_optional}')
        else:
            # Update mode: only copy what's present in src; destination must already exist
            if not scene_dst.exists():
                print(f'  [WARN]  {scene_label:<30} dst folder missing — run full mode first')
                continue

        if not dry_run:
            scene_dst.mkdir(parents=True, exist_ok=True)

        for band, src_file in band_files.items():
            # In update mode, remove old file with opposite case extension first
            # to prevent find_file() from picking the wrong one.
            if update_mode and not dry_run:
                for ext in ['.TIF', '.tif']:
                    old = scene_dst / (src_file.stem + ext)
                    if old.exists() and old != scene_dst / src_file.name:
                        old.unlink()
                        print(f'  [DEL]   removed old {old.name}')

            dst_file = scene_dst / src_file.name
            if dry_run:
                print(f"  [DRY]   {src_file.name} -> {dst_file}")
            else:
                shutil.copy2(src_file, dst_file)

        complete += 1
        print(f"  [OK]    {scene_label:<30} -> {scene_dst}")

    print(f"\n{'='*55}")
    print(f"Scenes processed: {complete}")
    if incomplete:
        print(f"Scenes skipped (missing required bands): {len(incomplete)}")
        for name, missing in incomplete:
            print(f"  {name}: missing {missing}")
    if not update_mode:
        print(f"\nNext: python driver.py")
    else:
        print(f"\nNext: verify B10 dimensions then python driver.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Organize GEE-exported Landsat 9 bands into driver.py-ready folder structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--src', default='ISRO_PS10_L9_v2',
                        help='Flat Drive download folder (default: ISRO_PS10_L9)')
    parser.add_argument('--dst', default='input',
                        help='Destination root (default: input/)')
    parser.add_argument('--update', action='store_true',
                        help='Update mode: copy only bands present in src, overwriting existing. '
                             'Use for partial re-exports (e.g. B10 only).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without copying files.')
    args = parser.parse_args()

    organize(Path(args.src), Path(args.dst),
             dry_run=args.dry_run, update_mode=args.update)
