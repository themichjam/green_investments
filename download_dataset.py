# download_dataset.py
import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Specify the dataset path on Kaggle
dataset = 'pritish509/s-and-p-500-esg-risk-ratings'

# Specify the path where you want to download the dataset
path = 'path/to/your/project/directory'

# Download the dataset
api.dataset_download_files(dataset, path=path, unzip=True)
