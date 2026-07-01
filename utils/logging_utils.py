import logging
import os

def setup_logging(log_name='pipeline', log_dir='output'):
    """
    Sets up logging to both a file in the specified log_dir and the console.
    """
    # Tolerant of accidental positional/path args: ensure dir exists and build
    # the filename from the basename only so a stray path can't malform it.
    os.makedirs(log_dir, exist_ok=True)
    safe_log_name = os.path.basename(str(log_name)) or 'pipeline'
    log_path = os.path.join(log_dir, f'{safe_log_name}.log')

    # Remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # encoding='utf-8' so non-ASCII log chars (→, ×) don't crash the
            # cp1252 default FileHandler on Windows.
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(safe_log_name)
