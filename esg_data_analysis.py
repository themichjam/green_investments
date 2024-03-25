import pandas as pd
import matplotlib.pyplot as plt

# Corrected file path to match the actual location
file_path = 'SP 500 ESG Risk Ratings.csv'
df = pd.read_csv(file_path)

# Understanding Your Data
print(df.head())
print(df.info())
print(df.describe())

# Cleaning the Data
# Fill missing values in 'ESG Risk Level' with 'Unknown'
df['ESG Risk Level'] = df['ESG Risk Level'].fillna('Unknown')
# Correcting the column name in the dropna method
df.dropna(subset=['Total ESG Risk score', 'ESG Risk Percentile'], inplace=True)
# Instead of 'ESG Score', use 'Total ESG Risk score'
df['Total ESG Risk score'] = pd.to_numeric(df['Total ESG Risk score'], errors='coerce')

# Analyzing ESG Scores
# Correcting the column name for histogram plotting
df['Total ESG Risk score'].hist(bins=20)
plt.title('Distribution of Total ESG Risk Scores')
plt.xlabel('Total ESG Risk Score')
plt.ylabel('Frequency')
plt.show()


# NOTE: The following analyses assume additional columns like 'Sector' exist.
# If such columns are not in your dataset, you'll need to adjust or skip these analyses.

# ESG Scores by Sector (if sector data is available)
# esg_scores_by_sector = df.groupby('Sector')['ESG Score'].mean()
# esg_scores_by_sector.plot(kind='bar')
# plt.title('Average ESG Score by Sector')
# plt.xlabel('Sector')
# plt.ylabel('Average ESG Score')
# plt.xticks(rotation=45)
# plt.show()

# Risk Level Analysis
risk_level_counts = df['ESG Risk Level'].value_counts()
risk_level_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('ESG Risk Level Distribution')
plt.ylabel('')
plt.show()

# Correlations (if financial performance metrics are available)
# print(df.corr())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
# Ensure 'Total ESG Risk score' and 'Sector' columns are correctly named and used
average_esg_by_sector = df.groupby('Sector')['Total ESG Risk score'].mean().sort_values()

plt.figure(figsize=(10, 8))
sns.barplot(x=average_esg_by_sector.values, y=average_esg_by_sector.index)
plt.title('Average ESG Score by Sector')
plt.xlabel('Average ESG Score')
plt.ylabel('Sector')
plt.show()

# Count of companies by ESG Risk Level
risk_level_counts = df['ESG Risk Level'].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 6))
risk_level_counts.plot(kind='bar')
plt.title('Distribution of Companies by ESG Risk Level')
plt.xlabel('ESG Risk Level')
plt.ylabel('Percentage of Companies')
plt.xticks(rotation=45)
plt.show()

# Pivot table to count risk levels by sector
risk_level_by_sector = pd.pivot_table(df, index='Sector', columns='ESG Risk Level', aggfunc='size', fill_value=0)

# Heatmap of ESG risk levels by sector
plt.figure(figsize=(12, 8))
sns.heatmap(risk_level_by_sector, annot=True, fmt="d", cmap='viridis')
plt.title('ESG Risk Levels by Sector')
plt.xlabel('ESG Risk Level')
plt.ylabel('Sector')
plt.show()

# Calculate the average Total ESG Risk score for each sector
sector_avg_esg = df.groupby('Sector')['Total ESG Risk score'].mean().sort_values()

# Print the average ESG scores by sector
print(sector_avg_esg)

# Identify the top 3 sectors with the highest ESG scores
print("Sectors with the highest average ESG scores:")
print(sector_avg_esg.tail(3))

# Identify the top 3 sectors with the lowest ESG scores
print("Sectors with the lowest average ESG scores:")
print(sector_avg_esg.head(3))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.barplot(y=sector_avg_esg.index, x=sector_avg_esg.values, palette='coolwarm')
plt.title('Average Total ESG Risk Score by Sector')
plt.xlabel('Average Total ESG Risk Score')
plt.ylabel('Sector')
plt.show()

# Calculate the distribution of ESG risk levels by sector
esg_risk_distribution = df.groupby(['Sector', 'ESG Risk Level']).size().unstack(fill_value=0)

# Normalize the distribution to get the percentage of companies in each risk level by sector
esg_risk_distribution_percent = esg_risk_distribution.div(esg_risk_distribution.sum(axis=1), axis=0) * 100

esg_risk_distribution_percent.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('ESG Risk Level Distribution by Sector')
plt.xlabel('Sector')
plt.ylabel('Percentage of Companies')
plt.legend(title='ESG Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import seaborn as sns

plt.figure(figsize=(14, 8))
sns.heatmap(esg_risk_distribution_percent, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('ESG Risk Level Distribution by Sector (%)')
plt.xlabel('ESG Risk Level')
plt.ylabel('Sector')
plt.show()

from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and it's already clean
X = df[['Environment Risk Score', 'Governance Risk Score', 'Social Risk Score']]  # Features
y = df['Total ESG Risk score']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Assuming average_esg_by_sector is a Series with index as sector names and values as average ESG scores
average_esg_by_sector.plot(kind='bar', figsize=(10, 6))
plt.title('Average Total ESG Risk Score by Sector')
plt.xlabel('Sector')
plt.ylabel('Average Total ESG Risk Score')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

risk_level_counts = df['ESG Risk Level'].value_counts()
risk_level_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, figsize=(8, 8), wedgeprops=dict(width=0.4))
plt.title('Distribution of Companies by ESG Risk Level')
plt.ylabel('')  # Remove the y-label since it's clear we're showing ESG Risk Levels
plt.show()

# Convert 'Full Time Employees' to numeric, assuming it's stored as a string
df['Full Time Employees'] = pd.to_numeric(df['Full Time Employees'].str.replace(',', ''), errors='coerce')

sns.scatterplot(data=df, x='Full Time Employees', y='Total ESG Risk score', alpha=0.5)
plt.title('Company Size vs. Total ESG Risk Score')
plt.xlabel('Full Time Employees')
plt.ylabel('Total ESG Risk Score')
plt.xscale('log')  # Use logarithmic scale if wide range of company sizes
plt.xlim(left=1)  # Adjust based on your data to avoid log(0) issues
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()

# Assuming esg_risk_distribution_percent is a DataFrame prepared earlier for the heatmap
sns.heatmap(esg_risk_distribution_percent, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Percentage Distribution of ESG Risk Levels by Sector')
plt.xlabel('ESG Risk Level')
plt.ylabel('Sector')
plt.show()
