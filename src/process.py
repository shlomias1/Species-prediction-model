from osgeo import gdal
import pandas as pd

# הוצאת חריגים
# sanity checkwith model

def pixel_to_geo(x_pixel, y_pixel, transform):
    x_geo = transform[0] + x_pixel * transform[1] + y_pixel * transform[2]
    y_geo = transform[3] + x_pixel * transform[4] + y_pixel * transform[5]
    return x_geo, y_geo

def geo_to_pixel(x_geo, y_geo, transform):
    col = int((x_geo - transform[0]) / transform[1])
    row = int((y_geo - transform[3]) / transform[5])
    return row, col

def get_pixel_values(tif_file_path, row, col):
    dataset = gdal.Open(tif_file_path)
    if not dataset:
        print(f"❌ Cannot open file: {tif_file_path}")
        return
    pixel_values = {}
    num_bands = dataset.RasterCount
    for i in range(1, num_bands + 1):
        band = dataset.GetRasterBand(i)
        pixel_value = band.ReadAsArray(col, row, 1, 1)[0][0]
        pixel_values[f"Band {i}"] = pixel_value
    dataset = None 
    return pixel_values

def tensor_to_bioclimatic_df(tensor, source=None):
    """
    ממיר טנזור בצורה (n_bio, n_year, n_month) ל־DataFrame עם עמודות year, month, pr, tas, tasmin, tasmax
    אם צריך, מבצע permute.
    """
    if tensor.shape[0] == 4 and tensor.shape[1] == 19 and tensor.shape[2] == 12:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape != (19, 12, 4):
        raise ValueError(f"שגיאה בצורת הטנזור: {tensor.shape}")

    years = list(range(2000, 2000 + tensor.shape[0]))
    months = list(range(1, 13))

    records = []
    for y_idx, year in enumerate(years):
        for m_idx, month in enumerate(months):
            pr, tas, tasmin, tasmax = tensor[y_idx, m_idx]
            record = {
                "year": year,
                "month": month,
                "pr": pr.item(),
                "tas": tas.item(),
                "tasmin": tasmin.item(),
                "tasmax": tasmax.item()
            }
            if source:
                record["file"] = source
            records.append(record)

    return pd.DataFrame(records)