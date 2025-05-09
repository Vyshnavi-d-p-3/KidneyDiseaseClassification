import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.
        
    Raises:
        ValueError: If the YAML file is empty.
        Exception: Any other exception while reading the file.

    Returns:
        ConfigBox: Parsed YAML content.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise BoxValueError("YAML content is empty")
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e 

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates multiple directories.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): Whether to log directory creation.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary to a JSON file.
    
    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a JSON file and returns it as a ConfigBox.
    
    Args:
        path (Path): Path to the JSON file.
    
    Returns:
        ConfigBox: Loaded data.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data to a binary file using joblib.
    
    Args:
        data (Any): Data to save.
        path (Path): Path to the binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file using joblib.
    
    Args:
        path (Path): Path to the binary file.
    
    Returns:
        Any: Loaded data.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Returns file size in KB.
    
    Args:
        path (Path): Path to the file.
    
    Returns:
        str: Approximate size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring: str, filename: str):
    """Decodes a base64 image string and saves it as a file."""
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)

def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """Encodes an image file to base64."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
