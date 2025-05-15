from datetime import datetime
import torch
import numpy as np
import os
from config import LOG_DIR
import zipfile
from pathlib import Path

def _create_log(log_msg, log_type, log_file = "logs.txt"):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_file)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as log:
        log.write(f'{log_type} : {log_msg} | {current_time} \n')

def _connect_cuda():
    log_type = "Warning" 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 
        log_msg = f"✅ Running on GPU: {gpu_name} | Memory: {gpu_memory:.2f} GB \n GPU Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB \n GPU Cached Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        _create_log(log_msg, log_type)
        print(log_msg)
    else:
        device = torch.device("cpu")
        log_msg = "Running on CPU - No GPU detected!"
        _create_log(log_msg, log_type)
        print(log_msg)
    return device

def _get_last_part(path):
    normalized_path = os.path.normpath(path)
    arr_paths = normalized_path.split("\\")
    last_part = arr_paths[-1]
    return last_part

def _extract_zip_file(zip_path, extract_to=None):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"File not found: {zip_path}")
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0] 
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"✅ Extracted to: {extract_to}")
    return extract_to

def _check_file_size(file_path):
    path = Path(file_path)
    print("File exists?", path.exists())
    print("Size:", path.stat().st_size, "bytes")
