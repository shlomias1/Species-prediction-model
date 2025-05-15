---

# Species Prediction Model

A machine learning model for predicting species distribution based on geospatial and environmental data.

## Overview

This project uses data from the GeoCLEF dataset to build a predictive model for determining the likelihood of species presence in various geographical regions. The goal is to assist ecological research and environmental monitoring through automated predictions.

## Repository Structure

```
Species-prediction-model/
├── data/                                      # Sample input datasets
├── models/                                    # Trained models and model utils
│ └── ...                                      # (Will be created during training)
├── src/                                       # Core Python scripts
│ ├── config.py                                # Existing
│ ├── utils/                                   # Util scripts
│ │ ├── __init__.py
│ │ ├── tif2cog.py
│ │ ├── utils.py
│ │ └── validate_cloud_optimized_geo.py
│ ├── data_preparation/                       # Data preparation scripts
│ │ ├── __init__.py
│ │ ├── generate_time_series_sample.py
│ │ ├── make_bounding_box.py
│ │ ├── merges_LandCover_hdfs.py
│ │ ├── merges_ASTER_elevation_tifs.py
│ │ ├── monthly_rasters_preparation.py
│ │ └── process.py                           # Cleanup/processing process
│ ├── __init__.py
│ ├── load_data.py                           # Cleanup and feature preparation
│ ├── dowland.py                           
│ ├── EDA.py                                 # Data analysis
│ ├── train.py                               # Training module (required to add)
│ ├── predict.py                             # Prediction module (required to add)
│ └── main.py                                # Main pipeline
├── requirements.txt                         # Dependencies (required to create)
└── README.md                                # Documentation
└── project_GeoCELF2025.sbatch               # Shell executable filen
```

## Requirements

* Python 3.7+
* pandas
* scikit-learn
* numpy
* matplotlib
* geopandas (if geospatial data is used)

Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

1. **Prepare your data** in the `data/` directory. Follow the format shown in the sample files.
2. **Preprocess** the data:

```bash
python src/preprocessing.py
```

3. **Train** the model:

```bash
python src/train.py
```

4. **Predict** using new data:

```bash
python src/predict.py --input data/new_observations.csv --output predictions.csv
```

## Example Use Case

Use this model to:

* Predict bird or plant species in new geographical locations.
* Support environmental conservation efforts with predictive insights.
* Assist in biodiversity research.

## License

This project is open-source and available under the [MIT License](LICENSE).

---
