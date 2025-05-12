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

def unzip(zip_path,extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"File successfully extracted to folder:\n{extract_to}")

def get_last_part(path):
    normalized_path = os.path.normpath(path)
    arr_paths = normalized_path.split("\\")
    last_part = arr_paths[-1]
    return last_part

def get_files(folder_path):
    files = os.listdir(folder_path) 
    ignored_extensions = config.IGNORED_EXTENSIONS
    valid_files = [f for f in files if not any(f.endswith(ext) for ext in ignored_extensions) and not os.path.isdir(os.path.join(folder_path, f))]
    df = pd.DataFrame(valid_files, columns=["file_name"])
    df["file_type"] = df["file_name"].apply(lambda x: x.split(".")[-1] if "." in x else "Unknown")
    df["file_path"] = df["file_name"].apply(lambda x: os.path.join(folder_path, x))
    df["data_type"] = get_last_part(folder_path)
    return df

def concat_dataframes(*dfs):
    valid_dfs = [df for df in dfs if isinstance(df, pd.DataFrame)]
    if not valid_dfs:
        print("❌ No DataFrame found for thread.")
        return pd.DataFrame()
    concatenated_df = pd.concat(valid_dfs, ignore_index=True)
    return concatenated_df

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
        print(f"שגיאה בקריאת הקובץ: {e}")
        return None

def load_pt_file_from_zip_folder(file_like):
    try:
        buffer = io.BytesIO(file_like.read())
        data = torch.load(buffer, map_location=torch.device('cpu'))
        return data
    except Exception as e:
        print(f"שגיאה בקריאת הקובץ: {e}")
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

# פיצול 80-20
# train data
# BioClimatic_Monthly_PA_cube_train = load_bioclimatic_tensor_folder("data/train_data/BioclimTimeSeries/cubes/GLC25_PA_train_bioclimatic_monthly") # pt files
# BioClimatic_Monthly_PO_cube_train = load_zip_to_dataframe("data/train_data/BioclimTimeSeries/cubes/GLC25_PO_train_bioclimatic_monthly.zip") # pt files on zip file
# BioClimatic_Monthly_PO_cube_train = load_bioclimatic_tensor_from_df(BioClimatic_Monthly_PO_cube_train)
# BioClimatic_Monthly_PA_value_train = load_csv_to_dataframe(config.BioClimatic_Monthly_PA_value_train)
# BioClimatic_Monthly_PO_value_train = load_csv_to_dataframe(config.BioClimatic_Monthly_PO_value_train)
# BioClimatic_Average_1981_2010_Rasters_train = load_folder_tif_to_dataframe(config.BioClimatic_Average_1981_2010_Rasters_train)
# Climatic_Monthly_2000_2019_Rasters_train = load_folder_tif_to_dataframe(config.Climatic_Monthly_2000_2019_Rasters_train)
# ASTER_Elevation_Rasters_train = extract_tif_info(config.ASTER_Elevation_Rasters_train)
# HumanFootprint_Rasters_train = load_folder_tif_to_dataframe(config.HumanFootprint_Rasters_train)
# LandCover_Rasters_PA_train = load_csv_to_dataframe(config.LandCover_Rasters_PA_train)
# LandCover_Rasters_PO_train = load_csv_to_dataframe(config.LandCover_Rasters_PO_train)
# LandCover_MODIS_Terra_Aqua_train = extract_tif_info(config.LandCover_MODIS_Terra_Aqua_train)
# Soilgrids_Rasters_train = load_folder_tif_to_dataframe(config.Soilgrids_Rasters_train)
# Soilgrids_Rasters_PA_train = load_csv_to_dataframe(config.Soilgrids_Rasters_PA_train)
# Soilgrids_Rasters_PA_train = load_csv_to_dataframe(config.Soilgrids_Rasters_PA_train)
# BioClimatic_Average_1981_2010_PA_Values_train = load_csv_to_dataframe(config.BioClimatic_Average_1981_2010_PA_Values_train)
# BioClimatic_Average_1981_2010_PO_Values_train = load_csv_to_dataframe(config.BioClimatic_Average_1981_2010_PO_Values_train)
# Elevation_PA_Values_train = load_csv_to_dataframe(config.Elevation_PA_Values_train)
# Elevation_PO_Values_train = load_csv_to_dataframe(config.Elevation_PO_Values_train)
# HumanFootprint_PA_Values_train = load_csv_to_dataframe(config.HumanFootprint_PA_Values_train)
# HumanFootprint_PO_Values_train = load_csv_to_dataframe(config.HumanFootprint_PO_Values_train)
# LandCover_PA_Values_train = load_csv_to_dataframe(config.LandCover_PA_Values_train)
# LandCover_PO_Values_train = load_csv_to_dataframe(config.LandCover_PO_Values_train)
# Soilgrids_PA_Values_train = load_csv_to_dataframe(config.Soilgrids_PA_Values_train)
# Soilgrids_PO_Values_train = load_csv_to_dataframe(config.Soilgrids_PO_Values_train)
# PA_SatellitePatches_NIR_train_paths = list_files_BFS(config.PA_SatellitePatches_NIR_train) # BFS tif files
# PA_SatellitePatches_RGB_train_paths = list_files_BFS(config.PA_SatellitePatches_RGB_train) # BFS tif files
# PO_SatellitePatches_NIR_train_paths = list_files_BFS(config.PO_SatellitePatches_NIR_train) # BFS tif files
# PO_SatellitePatches_RGB_train_paths = list_files_BFS(config.PO_SatellitePatches_RGB_train) # BFS tif files
# PA_SatellitePatches_NIR_train = load_df_tif_to_dataframe(PA_SatellitePatches_NIR_train_paths) 
# PA_SatellitePatches_RGB_train = load_df_tif_to_dataframe(PA_SatellitePatches_RGB_train_paths) 
# PO_SatellitePatches_NIR_train = load_df_tif_to_dataframe(PO_SatellitePatches_NIR_train_paths)
# PO_SatellitePatches_RGB_train = load_df_tif_to_dataframe(PO_SatellitePatches_RGB_train_paths) 
# SatelliteTimeSeries_train = load_many_csv_to_dataframe(config.SatelliteTimeSeries_train)
# landsat_SatelliteTimeSeries_train = load_zip_to_dataframe("data/train_data/GLC24-PO-train-landsat-time-series.zip") # pt files on zip file
# landsat_SatelliteTimeSeries_train = load_bioclimatic_tensor_from_df(landsat_SatelliteTimeSeries_train) 
# PA_metadata_train = load_csv_to_dataframe(config.PA_metadata_train)
# PO_metadata_train = load_csv_to_dataframe(config.PO_metadata_train)
# # test data
# BioClimatic_Monthly_PA_cube_test = "data/test_data/BioclimTimeSeries/cubes" # pt files
# BioClimatic_Monthly_PA_value_test = load_csv_to_dataframe(config.BioClimatic_Monthly_PA_value_test)
# BioClimatic_Average_1981_2010_Rasters_test = load_folder_tif_to_dataframe(config.BioClimatic_Average_1981_2010_Rasters_test)
# Climatic_Monthly_2000_2019_Rasters_test = load_folder_tif_to_dataframe(config.Climatic_Monthly_2000_2019_Rasters_test)
# ASTER_Elevation_Rasters_test = extract_tif_info(config.ASTER_Elevation_Rasters_test)
# HumanFootprint_Rasters_test = load_folder_tif_to_dataframe(config.HumanFootprint_Rasters_test)
# LandCover_Rasters_PA_test = load_csv_to_dataframe(config.LandCover_Rasters_PA_test)
# Soilgrids_Rasters_test = load_folder_tif_to_dataframe(config.Soilgrids_Rasters_test)
# Soilgrids_Rasters_PA_test = load_csv_to_dataframe(config.Soilgrids_Rasters_PA_test)
# BioClimatic_Average_1981_2010_PA_Values_test = load_csv_to_dataframe(config.BioClimatic_Average_1981_2010_PA_Values_test)
# Elevation_PA_Values_test = load_csv_to_dataframe(config.Elevation_PA_Values_test)
# HumanFootprint_PA_Values_test = load_csv_to_dataframe(config.HumanFootprint_PA_Values_test)
# LandCover_PA_Values_test = load_csv_to_dataframe(config.LandCover_PA_Values_test)
# Soilgrids_PA_Values_test = load_csv_to_dataframe(config.Soilgrids_PA_Values_test)
# PA_SatellitePatches_NIR_test_paths = list_files_BFS(config.PA_SatellitePatches_NIR_test) # BFS tif files
# PA_SatellitePatches_RGB_test_paths = list_files_BFS(config.PA_SatellitePatches_RGB_test) # BFS tif files
# PA_SatellitePatches_NIR_test = load_df_tif_to_dataframe(PA_SatellitePatches_NIR_test_paths)
# PA_SatellitePatches_RGB_test = load_df_tif_to_dataframe(PA_SatellitePatches_RGB_test_paths)
# SatelliteTimeSeries_test = load_many_csv_to_dataframe(config.SatelliteTimeSeries_test)
# landsat_SatelliteTimeSeries_test = "data/test_data/GLC24-PA-test-landsat-time-series.zip" # zip file
# PA_metadata_test = load_csv_to_dataframe(config.PA_metadata_test)
