import os
import glob

def validate_extension(file_path, expected_extension=".TIF"):
    """Validates that a file has the expected extension (case-insensitive).

    Raises ValueError if the extension does not match. Callers discard the
    return value, so raising is the only way an invalid extension is surfaced.
    """
    if not file_path.lower().endswith(expected_extension.lower()):
        raise ValueError(f"Invalid file extension for {file_path}: expected {expected_extension}")

def find_file(directory, pattern):
    """Finds a file in the directory matching the pattern. Returns the first match or None."""
    # Add wildcards if not present to perform a 'contains' search
    search_pattern = pattern
    if not search_pattern.startswith('*'):
        search_pattern = '*' + search_pattern
    if not search_pattern.endswith('*'):
        search_pattern = search_pattern + '*'

    files = glob.glob(os.path.join(directory, search_pattern))
    # Prioritize .TIF (case-insensitive)
    tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]

    if tif_files:
        return tif_files[0]
    return files[0] if files else None
