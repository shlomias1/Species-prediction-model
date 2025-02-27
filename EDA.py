import load_data
import pandas as pd

PA_train_data = load_data.load_data
print(f"head: \n{PA_train_data.head()}")
print(f"shape: {PA_train_data.shape}")
print(f"columns: {PA_train_data.columns}")
print({PA_train_data.info()})