# analyze_data.py
import pandas as pd
import os

# Adjust the filename as necessary
filename = os.path.join('path/to/your/project/directory', 'YourDatasetFileName.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(filename)

# Now you can work with your DataFrame
print(df.head())
