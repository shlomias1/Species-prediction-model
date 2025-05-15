import os
import pandas as pd
import config
import load_data
import process
import EDA
from utils.utils import _create_log

def pipeline():
    load_data.extract_all_zip_files()
    
if __name__ == "__main__":
    pipeline()
