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

def load_env():
    """Loads environment variables from .env file at project root."""
    root = get_project_root()
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().strip('"').strip("'")
                        os.environ[key] = val

# Automatically load environment variables on import
load_env()
