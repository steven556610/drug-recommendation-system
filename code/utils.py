import logging
import sys
import os

def get_logger(name="BioRec"):
    """
    Creates a standard, beautifully formatted console logger for tracking system operations.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        # Professional standard timestamp formatting
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
    return logger

def get_project_root():
    """
    Returns the absolute path to the drug_rec_system root directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
