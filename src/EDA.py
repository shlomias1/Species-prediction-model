import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def summary(df):
    print(f"head: \n{df.head()}")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns}")
    print({df.info()})
    print(f"missing value each columns: \n{df.isnull().sum()}")
    print(f"statictic: \n{df.describe()}")

def check_categorial_columns(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        print(f"🔹 Column {col} - Number of unique categories: {df[col].nunique()}")
        print(df[col].value_counts().head(10)) 
        print("\n")

def check_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        print(f"🔹 Column {col} - Number of unique values: {df[col].nunique()}")
        print(f"Min: {df[col].min()} - Max: {df[col].max()}")
        print("\n")
           
PA_train_data = load_data.load_data
summary(PA_train_data)
check_categorial_columns(PA_train_data)
check_numeric_columns(PA_train_data)
