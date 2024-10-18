import os
import logging
from typing import Optional

def setup_output_path(custom_path: Optional[str] = None) -> str:
    """Set up and create the output directory."""
    if custom_path:
        output_path = custom_path
    else:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    os.makedirs(output_path, exist_ok=True)
    return output_path

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Function to set up a logger."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def save_metadata(samples: list, output_file: str):
    """Save metadata to a CSV file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(f"{sample[0]}|{sample[1]}|{sample[2] or ''}\n")
