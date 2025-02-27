import os
import pandas as pd
import config

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

def add_image_path_column(df, survey_id_column, new_column_name, base_image_path):
    df[survey_id_column] = df[survey_id_column].astype(int).astype(str)
    df[new_column_name] = (
        base_image_path + "/" +
        df[survey_id_column].str[-2:] + "/" +
        df[survey_id_column].str[-4:-2] + "/" +
        df[survey_id_column] + ".jpeg"
    )
    return df

BioClimatic_Average_1981_2010_path = os.path.join(config.BASE_DIR, "Climate", "BioClimatic_Average_1981-2010")
Climatic_Monthly_2000_2019_path = os.path.join(config.BASE_DIR, "Climate", "Climatic_Monthly_2000-2019")
Elevation_path = os.path.join(config.BASE_DIR, "Elevation")
LandCover_path = os.path.join(config.BASE_DIR, "LandCover")
HumanFootprint_detailed_path = os.path.join(config.BASE_DIR, "HumanFootprint", "detailed")
HumanFootprint_summarized_path = os.path.join(config.BASE_DIR, "HumanFootprint", "summarized")
Soilgrids_path = os.path.join(config.BASE_DIR, "Soilgrids")
PA_path = os.path.join(config.BASE_DIR, "GLC24-PA-metadata-train.csv")
PO_path = os.path.join(config.BASE_DIR, "GLC24_P0_metadata_train.csv")
SatelliteTimeSeries_path = os.path.join(config.BASE_DIR, "SatelliteTimeSeries")
PA_SatellitePatches_NIR_path = os.path.join(config.BASE_DIR, "PA-Train-SatellitePatches-NIR")
PA_SatellitePatches_RGB_path = os.path.join(config.BASE_DIR, "PA-Train-SatellitePatches-RGB")
PO_SatellitePatches_NIR_path = os.path.join(config.BASE_DIR, "PO-Train-SatellitePatches-NIR")
PO_SatellitePatches_RGB_path = os.path.join(config.BASE_DIR, "PO-Train-SatellitePatches-RGB")

PA_train_data = load_csv_to_dataframe(PA_path)
PA_train_data = add_image_path_column(PA_train_data, "surveyId", "SatellitePatches-NIR_path", PA_SatellitePatches_NIR_path)
PA_train_data = add_image_path_column(PA_train_data, "surveyId", "SatellitePatches-RGB_path", PA_SatellitePatches_RGB_path)

PO_train_data = load_csv_to_dataframe(PO_path)
PO_train_data = add_image_path_column(PO_train_data, "surveyId", "SatellitePatches-NIR_path", PO_SatellitePatches_NIR_path)
PO_train_data = add_image_path_column(PO_train_data, "surveyId", "SatellitePatches-RGB_path", PO_SatellitePatches_RGB_path)
