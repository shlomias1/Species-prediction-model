import os
import glob
import pandas as pd
import numpy as np
import config
import zipfile
from osgeo import gdal
import torch
from process import tensor_to_bioclimatic_df
import io
import zipfile
from pathlib import Path
import utils 

def get_files(folder_path):
    files = os.listdir(folder_path) 
    ignored_extensions = config.IGNORED_EXTENSIONS
    valid_files = [f for f in files if not any(f.endswith(ext) for ext in ignored_extensions) and not os.path.isdir(os.path.join(folder_path, f))]
    df = pd.DataFrame(valid_files, columns=["file_name"])
    df["file_type"] = df["file_name"].apply(lambda x: x.split(".")[-1] if "." in x else "Unknown")
    df["file_path"] = df["file_name"].apply(lambda x: os.path.join(folder_path, x))
    df["data_type"] = utils._get_last_part(folder_path)
    return df

def load_csv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path) 
        return df
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        return pd.DataFrame() 

def load_many_csv_to_dataframe(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []  
    bands = ['green', 'red', 'blue', 'nir', 'swir1', 'swir2']
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        region = "PA" if "PA" in filename else "PO" if "PO" in filename else "Unknown"
        band = next((b for b in bands if b in filename.lower()), "Unknown")
        df['region'] = region
        df['band'] = band
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def add_image_path_column(df, survey_id_column, new_column_name, base_image_path):
    df[survey_id_column] = df[survey_id_column].astype(int).astype(str)
    df[new_column_name] = (
        base_image_path + "/" +
        df[survey_id_column].str[-2:] + "/" +
        df[survey_id_column].str[-4:-2] + "/" +
        df[survey_id_column] + ".jpeg"
    )
    return df

def list_files_BFS(folder_path):
    files_data = []
    queue = [folder_path]
    while queue:
        current_path = queue.pop(0)
        for entry in os.listdir(current_path):
            full_path = os.path.join(current_path, entry)
            if os.path.isdir(full_path):
                queue.append(full_path)
            else:
                files_data.append((entry, full_path))
    df = pd.DataFrame(files_data, columns=["file_name", "file_path"])
    return df

def load_zip_to_dataframe(zip_path):
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} אינו קובץ ZIP תקני.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_data = []
        for file_path in zip_ref.namelist():
            if file_path.endswith("/"):
                continue
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower().replace('.', '')
            file_data.append({
                'zip_path': zip_path,
                'inner_path': file_path,
                'file_name': file_name,
                'file_type': file_ext or 'unknown'
            })
    return pd.DataFrame(file_data)

def load_pt_file(pt_path):
    try:
        data = torch.load(pt_path, map_location=torch.device('cpu'))
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def load_pt_file_from_zip_folder(file_like):
    try:
        buffer = io.BytesIO(file_like.read())
        data = torch.load(buffer, map_location=torch.device('cpu'))
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def load_bioclimatic_tensor_folder(folder_path):
    all_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pt"):
            path = os.path.join(folder_path, file)
            try:
                tensor = load_pt_file(path)
                df = tensor_to_bioclimatic_df(tensor, source=file)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error in file {file}: {e}") 
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def load_bioclimatic_tensor_from_df(df_files, zip_column='zip_path', inner_column='inner_path'):
    all_dfs = []
    for idx, row in df_files.iterrows():
        zip_path = row[zip_column]
        inner_path = row[inner_column]
        file_name = os.path.basename(inner_path)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(inner_path) as file:
                    tensor = load_pt_file_from_zip_folder(file) 
                    df = tensor_to_bioclimatic_df(tensor, source=file_name)
                    all_dfs.append(df)
        except Exception as e:
            print(f"Error in file {inner_path} in {zip_path}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def extract_tif_info(tif_file_path):
    try:
        dataset = gdal.Open(tif_file_path)
        if not dataset:
            print(f"❌ Cannot open file: {tif_file_path}")
            return
        print(f"✅ File successfully loaded: {tif_file_path}")
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        num_bands = dataset.RasterCount
        projection = dataset.GetProjection()
        transform = dataset.GetGeoTransform()
        data_for_band = []
        for i in range(1, num_bands + 1):
            band = dataset.GetRasterBand(i)
            stats = band.GetStatistics(True, True)
            array = band.ReadAsArray()
            if band.GetNoDataValue() is not None:
                array = array[array != band.GetNoDataValue()]
            median = float(np.median(array))
            data_for_band.append({
                f"Band {i}": {
                    "Min": stats[0],
                    "Max": stats[1],
                    "Avg": stats[2],
                    "Std": stats[3],
                    "Median": median
                }
            })
        dataset = None
        return {
            "tif file path": tif_file_path,
            "width": width,
            "height": height,
            "num_bands": num_bands,
            "projection": projection,
            "transform_matrix": transform,
            "data_for_band": data_for_band
        }
    except FileNotFoundError:
        print(f"❌ File not found: {tif_file_path}")
    except Exception as e:
        print(f"❌ Error ({type(e).__name__}): {e}")
        
def load_folder_tif_to_dataframe(folder_path):
    tif_paths = glob.glob(os.path.join(folder_path, "*.tif")) + \
                glob.glob(os.path.join(folder_path, "*.tiff"))
    all_data = []
    for tif_path in tif_paths:
        info = extract_tif_info(tif_path)
        if info:
            row_data = {
                "file_path": tif_path,
                "width": info["width"],
                "height": info["height"],
                "num_bands": info["num_bands"],
                "projection": info["projection"],
                "transform_matrix": info["transform_matrix"],
                "data_for_band": info["data_for_band"]
            }
            all_data.append(row_data)

    result_df = pd.DataFrame(all_data)
    return result_df

def load_df_tif_to_dataframe(df):
    all_data = []
    for _, row in df.iterrows():
        tif_path = row['file_path']
        info = extract_tif_info(tif_path)
        if info:
            row_data = {
                "file_path": tif_path,
                "width": info["width"],
                "height": info["height"],
                "num_bands": info["num_bands"],
                "projection": info["projection"],
                "transform_matrix": info["transform_matrix"],
                "data_for_band": info["data_for_band"]
            }
            all_data.append(row_data)
    result_df = pd.DataFrame(all_data)
    return result_df

def extract_all_zip_files():
    utils._extract_zip_file(config.EnvironmentalRasters_Climate)
    utils._extract_zip_file(config.EnvironmentalRasters_Elevation)
    utils._extract_zip_file(config.EnvironmentalRasters_HumanFootprint)
    utils._extract_zip_file(config.EnvironmentalRasters_LandCover)
    utils._extract_zip_file(config.EnvironmentalRasters_Soilgrids)
    utils._extract_zip_file(config.EnvironmentalValues)
    utils._extract_zip_file(config.SatellitePatches_PA_NIR_Train)
    utils._extract_zip_file(config.SatellitePatches_PA_RGB_Train)
    utils._extract_zip_file(config.SatellitePatches_PO_NIR_Train)
    utils._extract_zip_file(config.SatellitePatches_PO_RGB_Train)
    utils._extract_zip_file(config.SatelliteTimeSeries_PA_cube_Train)
    utils._extract_zip_file(config.SatelliteTimeSeries_PO_cube_Train)
    utils._extract_zip_file(config.BioClimaticTimeSeries_cube_PA_Test)
    utils._extract_zip_file(config.SatellitePatches_PA_NIR_Test)
    utils._extract_zip_file(config.SatellitePatches_PA_RGB_Test)
    utils._extract_zip_file(config.SatelliteTimeSeries_PA_cube_Test)

# Train
# # BioClimaticTimeSeries cube
# BioClimaticTimeSeries_cube_PA_Train = load_zip_to_dataframe(config.BioClimaticTimeSeries_cube_PA_Train)
# BioClimaticTimeSeries_cube_PO_Train = load_zip_to_dataframe(config.BioClimaticTimeSeries_cube_PO_Train)

# # BioClimaticTimeSeries value
# BioClimaticTimeSeries_value_PA_Train = load_csv_to_dataframe(config.BioClimaticTimeSeries_value_PA_Train) 
# BioClimaticTimeSeries_value_PO_Train = load_csv_to_dataframe(config.BioClimaticTimeSeries_value_PO_Train) 

# # EnvironmentalRasters Climate
# EnvironmentalRasters_Climate_PA_Train_Average = load_csv_to_dataframe(config.EnvironmentalRasters_Climate_PA_Train_Average) 
# EnvironmentalRasters_Climate_PA_Train_Monthly = load_csv_to_dataframe(config.EnvironmentalRasters_Climate_PA_Train_Monthly) 
# EnvironmentalRasters_Climate_PO_Train_Average = load_csv_to_dataframe(config.EnvironmentalRasters_Climate_PO_Train_Average) 
# EnvironmentalRasters_Climate_PO_Train_Monthly = load_csv_to_dataframe(config.EnvironmentalRasters_Climate_PO_Train_Monthly) 
# EnvironmentalRasters_Climate = load_zip_to_dataframe(config.EnvironmentalRasters_Climate)

# # EnvironmentalRasters Elevation
# EnvironmentalRasters_Elevation_PA_Train = load_csv_to_dataframe(config.EnvironmentalRasters_Elevation_PA_Train) 
# EnvironmentalRasters_Elevation_PO_Train = load_csv_to_dataframe(config.EnvironmentalRasters_Elevation_PO_Train) 
# EnvironmentalRasters_Elevation = load_zip_to_dataframe(config.EnvironmentalRasters_Elevation)

# # EnvironmentalRasters HumanFootprint
# EnvironmentalRasters_HumanFootprint_PA_Train = load_csv_to_dataframe(config.EnvironmentalRasters_HumanFootprint_PA_Train) 
# EnvironmentalRasters_HumanFootprint_PO_Train = load_csv_to_dataframe(config.EnvironmentalRasters_HumanFootprint_PO_Train) 
# EnvironmentalRasters_HumanFootprint = load_zip_to_dataframe(config.EnvironmentalRasters_HumanFootprint)

# # EnvironmentalRasters LandCover
# EnvironmentalRasters_LandCover_PA_Train = load_csv_to_dataframe(config.EnvironmentalRasters_LandCover_PA_Train) 
# EnvironmentalRasters_LandCover_PO_Train = load_csv_to_dataframe(config.EnvironmentalRasters_LandCover_PO_Train) 
# EnvironmentalRasters_LandCover = load_zip_to_dataframe(config.EnvironmentalRasters_LandCover)

# # EnvironmentalRasters Soilgrids
# EnvironmentalRasters_Soilgrids_PA_Train = load_csv_to_dataframe(config.EnvironmentalRasters_Soilgrids_PA_Train) 
# EnvironmentalRasters_Soilgrids_PO_Train = load_csv_to_dataframe(config.EnvironmentalRasters_Soilgrids_PO_Train) 
# EnvironmentalRasters_Soilgrids = load_zip_to_dataframe(config.EnvironmentalRasters_Soilgrids)


# # EnvironmentalValues
# EnvironmentalValues = load_zip_to_dataframe(config.EnvironmentalValues)

# # PresenceAbsenceSurveys
# PA_metadata_Train = load_csv_to_dataframe(config.PA_metadata_Train) 

# # PresenceOnlyOccurrences
# PO_metadata_Train = load_csv_to_dataframe(config.PO_metadata_Train) 

# # SatellitePatches
# SatellitePatches_PA_NIR_Train = load_zip_to_dataframe(config.SatellitePatches_PA_NIR_Train)
# SatellitePatches_PA_RGB_Train = load_zip_to_dataframe(config.SatellitePatches_PA_RGB_Train)
# SatellitePatches_PO_NIR_Train = load_zip_to_dataframe(config.SatellitePatches_PO_NIR_Train)
# SatellitePatches_PO_RGB_Train = load_zip_to_dataframe(config.SatellitePatches_PO_RGB_Train)

# # SatelliteTimeSeries values
# SatelliteTimeSeries_PA_Blue_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_Blue_Train) 
# SatelliteTimeSeries_PA_Green_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_Green_Train) 
# SatelliteTimeSeries_PA_NIR_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_NIR_Train) 
# SatelliteTimeSeries_PA_Red_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_Red_Train) 
# SatelliteTimeSeries_PA_SWIR1_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_SWIR1_Train) 
# SatelliteTimeSeries_PA_SWIR2_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_SWIR2_Train) 
# SatelliteTimeSeries_PO_Blue_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PO_Blue_Train) 
# SatelliteTimeSeries_PO_Green_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PO_Green_Train) 
# SatelliteTimeSeries_PO_NIR_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PO_NIR_Train) 
# SatelliteTimeSeries_PO_Red_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PO_Red_Train) 
# SatelliteTimeSeries_PO_SWIR1_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PO_SWIR1_Train) 
# SatelliteTimeSeries_PO_SWIR2_Train = load_csv_to_dataframe(config.SatelliteTimeSeries_PO_SWIR2_Train) 

# # SatelliteTimeSeries cube
# SatelliteTimeSeries_PA_cube_Train = load_zip_to_dataframe(config.SatelliteTimeSeries_PA_cube_Train)
# SatelliteTimeSeries_PO_cube_Train = load_zip_to_dataframe(config.SatelliteTimeSeries_PO_cube_Train)

# # Test
# # BioClimaticTimeSeries cube
# BioClimaticTimeSeries_cube_PA_Test = load_zip_to_dataframe(config.BioClimaticTimeSeries_cube_PA_Test)

# # BioClimaticTimeSeries value
# BioClimaticTimeSeries_value_PA_Test = load_csv_to_dataframe(config.BioClimaticTimeSeries_value_PA_Test) 

# # EnvironmentalRasters Climate
# EnvironmentalRasters_Climate_PA_Test_Average = load_csv_to_dataframe(config.EnvironmentalRasters_Climate_PA_Test_Average) 
# EnvironmentalRasters_Climate_PA_Test_Monthly = load_csv_to_dataframe(config.EnvironmentalRasters_Climate_PA_Test_Monthly) 

# # EnvironmentalRasters Elevation
# EnvironmentalRasters_Elevation_PA_Test = load_csv_to_dataframe(config.EnvironmentalRasters_Elevation_PA_Test) 

# # EnvironmentalRasters HumanFootprint
# EnvironmentalRasters_HumanFootprint_PA_Test = load_csv_to_dataframe(config.EnvironmentalRasters_HumanFootprint_PA_Test) 

# # EnvironmentalRasters LandCover
# EnvironmentalRasters_LandCover_PA_Test = load_csv_to_dataframe(config.EnvironmentalRasters_LandCover_PA_Test) 

# # EnvironmentalRasters Soilgrids
# EnvironmentalRasters_Soilgrids_PA_Test = load_csv_to_dataframe(config.EnvironmentalRasters_Soilgrids_PA_Test) 

# # EnvironmentalValues

# # PresenceAbsenceSurveys
# PA_metadata_Test = load_csv_to_dataframe(config.PA_metadata_Test) 

# # PresenceOnlyOccurrences

# # SatellitePatches
# SatellitePatches_PA_NIR_Test = load_zip_to_dataframe(config.SatellitePatches_PA_NIR_Test)
# SatellitePatches_PA_RGB_Test = load_zip_to_dataframe(config.SatellitePatches_PA_RGB_Test)

# # SatelliteTimeSeries values
# SatelliteTimeSeries_PA_Blue_Test = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_Blue_Test) 
# SatelliteTimeSeries_PA_Green_Test = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_Green_Test) 
# SatelliteTimeSeries_PA_NIR_Test = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_NIR_Test) 
# SatelliteTimeSeries_PA_Red_Test = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_Red_Test) 
# SatelliteTimeSeries_PA_SWIR1_Test = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_SWIR1_Test) 
# SatelliteTimeSeries_PA_SWIR2_Test = load_csv_to_dataframe(config.SatelliteTimeSeries_PA_SWIR2_Test) 

# # SatelliteTimeSeries cube
# SatelliteTimeSeries_PA_cube_Test = load_zip_to_dataframe(config.SatelliteTimeSeries_PA_cube_Test)
