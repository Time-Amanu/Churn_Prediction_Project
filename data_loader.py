import pandas as pd
import zipfile
import io
import os

def load_and_clean_data(file_path):
    # Simplified logic for the repo utility
    df = pd.read_csv(file_path)
    df = df.dropna().drop(columns=['CustomerID'], errors='ignore')
    return df