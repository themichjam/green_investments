import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Specify the dataset identifier for "S&P 500 ESG Risk Ratings"
dataset_identifier = 'pritish509/s-and-p-500-esg-risk-ratings'

# Download dataset files to a temporary directory named 'temp_kaggle_data' and unzip them
api.dataset_download_files(dataset_identifier, path='temp_kaggle_data', unzip=True)

# List files in the temp directory to confirm the file name
temp_dir = 'temp_kaggle_data'
files = os.listdir(temp_dir)
print("Files in temp directory:", files)

# Correct the file name based on the actual file present in the directory
# This name was corrected based on the output of the script indicating the actual downloaded file's name
file_path = os.path.join('temp_kaggle_data', 'SP 500 ESG Risk Ratings.csv')

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Optional: Cleanup - Remove the temporary directory after use
shutil.rmtree('temp_kaggle_data')
