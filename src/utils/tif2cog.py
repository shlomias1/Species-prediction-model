import os
import shutil
import subprocess
import tempfile
from validate_cloud_optimized_geotiff import validate

def tif2cog(input_file_list, output_file, levels=6, num_threads="ALL_CPUS", cachemax=4096, tmp_folder=None):
    """Cloud Optimized GeoTIFF from a list of GeoTIFFs"
    Use gdal functions to create Cloud Optimized GeoTIFF from a list
    of GeoTIFFs.
    :param input_file_list: input file list to be use with gdalbuildvrt.
    :param output_file: output file name (cloud optimized geotiff).
    :param levels: number of overviews to build. Default to 6: 2, 4,
        8, 16, 32, 64.
    :param num_threads: value for GDAL_NUM_THREADS option.
    :param cachemax: value for GDAL_CACHEMAX option.
    :return: A tuple, whose first element is an array of error
    messages (empty if there is no error), and the second element, a
    dictionary with the structure of the GeoTIFF file.

    """
    # Attributes os.getcwd() to tmp_folder is not provided
    tmp_folder = tmp_folder if tmp_folder is not None else os.getcwd()
    tmp_dir = tempfile.mkdtemp(dir=tmp_folder)
    print("Created temp directory " + tmp_dir)
    # Step 1: Mosaicing
    print("Mosaicing with gdalbuildvrt")
    gdal_cmd = ["gdalbuildvrt",
                "-input_file_list", input_file_list,
                os.path.join(tmp_dir, "virtualfile.vrt")]
    subprocess.call(" ".join(gdal_cmd), shell=True)
    # Step 2: Create big Geotiff from vrt
    print("Creating big Geotiff from vrt")
    gdal_cmd = ["gdal_translate", # "-ot Int16",
                "-of GTiff",
                "-co BIGTIFF=YES",
                "-co TILED=YES",
                "-co COMPRESS=DEFLATE",
                "-co PREDICTOR=2",
                "-co NUM_THREADS=" + str(num_threads),
                "--config GDAL_CACHEMAX " + str(cachemax),
                os.path.join(tmp_dir, "virtualfile.vrt"),
                os.path.join(tmp_dir, "bigtif.tif")]
    subprocess.call(" ".join(gdal_cmd), shell=True)
    # Step 3: Build overviews
    print("Building overviews")
    file_name = os.path.join(tmp_dir, "bigtif.tif")
    for i in range(levels):
        ov = pow(2, i+1)
        print("Compute overview: " + str(ov))
        gdal_cmd = ["gdaladdo",
                    "--config GDAL_NUM_THREADS " + str(num_threads),
                    "--config GDAL_CACHEMAX " + str(cachemax),
                    "--config COMPRESS_OVERVIEW DEFLATE",
                    "--config PREDICTOR_OVERVIEW 2",
                    "--config BIGTIFF_OVERVIEW IF_SAFER",
                    "-ro",
                    "-r near " + file_name + " 2"]
        subprocess.call(" ".join(gdal_cmd), shell=True)
        file_name = file_name + ".ovr"
    # Step 4: Create Cloud Optimized Geotiff (COG)
    print("Creating Cloud Optimized Geotiff (COG)")
    gdal_cmd = ["gdal_translate",
                "-of GTiff",
                "-co BIGTIFF=YES",
                "-co TILED=YES",
                "-co BLOCKXSIZE=256",
                "-co BLOCKYSIZE=256",
                "-co COMPRESS=DEFLATE",
                "-co PREDICTOR=2",
                "-co COPY_SRC_OVERVIEWS=YES",
                "-co NUM_THREADS=" + str(num_threads),
                "--config GDAL_TIFF_OVR_BLOCKSIZE 128",
                "--config GDAL_CACHEMAX " + str(cachemax),
                os.path.join(tmp_dir, "bigtif.tif"),
                output_file]
    subprocess.call(" ".join(gdal_cmd), shell=True)
    # Remove temporary folder
    shutil.rmtree(tmp_dir)  # delete directory
    print("Deleted temp directory " + tmp_dir)
    # Step 4: Validate COG
    print("Validating COG")
    #check = validate_cloud_optimized_geotiff.validate(output_file)
    check = validate(output_file)
    return check