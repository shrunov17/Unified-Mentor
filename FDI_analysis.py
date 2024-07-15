# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:44:33 2024

@author: DELL
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
file_path = r'D:\IIT DELHI Material\Prrrrooojjjecccttt\unified mentor\FDI data.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Reshape the data to have 'Year' as a column
data = pd.melt(data, id_vars=['Sector'], var_name='Year', value_name='FDI Inflows')
data['Year'] = data['Year'].apply(lambda x: int(x.split('-')[0]) + 1)  # Convert fiscal year to calendar year

# Ensure the 'Year' column is treated as a numeric type
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Fit regression models to each sector
sectors = data['Sector'].unique()
regression_results = []

for sector in sectors:
    sector_data = data[data['Sector'] == sector]
    X = sector_data['Year'].values.reshape(-1, 1)
    y = sector_data['FDI Inflows'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    regression_results.append((sector, slope, intercept, sector_data))

# Identify sectors with the most stable and increasing slopes
regression_results.sort(key=lambda x: x[1])
stable_sectors = regression_results[:5]
increasing_sectors = regression_results[-5:]

# Plot time series with regression lines for top 5 sectors with increasing slopes
plt.figure(figsize=(14, 8))
for i, (sector, slope, intercept, sector_data) in enumerate(increasing_sectors):
    # plt.subplot(2, 3, i+1)
    plt.figure(figsize=(14, 8))
    plt.plot(sector_data['Year'], sector_data['FDI Inflows'], marker='o', label='FDI Inflows')
    plt.plot(sector_data['Year'], intercept + slope * sector_data['Year'], color='red', label='Regression Line')
    plt.title(sector)
    plt.xlabel('Year')
    plt.ylabel('FDI Inflows')
    plt.legend()
    plt.text(sector_data['Year'].max(), sector_data['FDI Inflows'].max(), f"Mean: {sector_data['FDI Inflows'].mean():.2f}\nMedian: {sector_data['FDI Inflows'].median():.2f}\nVariance: {sector_data['FDI Inflows'].var():.2f}", fontsize=12)
plt.tight_layout()
plt.show()

# Plot time series with regression lines for sectors with flat or decreasing slopes
plt.figure(figsize=(14, 8))
for i, (sector, slope, intercept, sector_data) in enumerate(stable_sectors):
    # plt.subplot(2, 3, i+1)
    plt.figure(figsize=(14, 8))
    plt.plot(sector_data['Year'], sector_data['FDI Inflows'], marker='o', label='FDI Inflows')
    plt.plot(sector_data['Year'], intercept + slope * sector_data['Year'], color='red', label='Regression Line')
    plt.title(sector)
    plt.xlabel('Year')
    plt.ylabel('FDI Inflows')
    plt.legend()
    plt.text(sector_data['Year'].max(), sector_data['FDI Inflows'].max(), f"Mean: {sector_data['FDI Inflows'].mean():.2f}\nMedian: {sector_data['FDI Inflows'].median():.2f}\nVariance: {sector_data['FDI Inflows'].var():.2f}", fontsize=12)
plt.tight_layout()
plt.show()

#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pivot the data to have sectors as columns and years as index
sector_pivot = data.pivot(index='Year', columns='Sector', values='FDI Inflows')
correlation_matrix = sector_pivot.corr()

# Unstack the correlation matrix and convert it to a DataFrame
correlation_pairs = correlation_matrix.unstack()
correlation_pairs = pd.DataFrame(correlation_pairs)
index_list = correlation_pairs.index.to_list()
# Example lists
sector1 = [index_list[i][0] for i in range(len(index_list))]
sector2 = [index_list[i][1] for i in range(len(index_list))]
corr_values = correlation_pairs[0].to_list()

# Create a DataFrame
correlation_pairs = pd.DataFrame({
    'Sector1': sector1,
    'Sector2': sector2,
    'Correlation': corr_values
})

# Display the DataFrame
# print(correlation_pairs)

# correlation_pairs = correlation_pairs.rename(columns={'Correlation': 'Correlation', 'Sector': 'Sector1', 'Sector': 'Sector2'})

# Filter out self-correlation and drop duplicate pairs
correlation_pairs = correlation_pairs[correlation_pairs['Sector1'] != correlation_pairs['Sector2']]
correlation_pairs = correlation_pairs.drop_duplicates(subset=['Sector1', 'Sector2'])

# Identify the top 5 pairs of sectors with the highest correlation
top_correlated_pairs = correlation_pairs.sort_values(by='Correlation', ascending=False).head(5)

print(top_correlated_pairs)

# Extract the unique sectors from the top correlated pairs
top_sectors = pd.unique(top_correlated_pairs[['Sector1', 'Sector2']].values.ravel('K'))

# Display the correlation matrix for the top sectors
top_sector_pivot = sector_pivot[top_sectors]
top_correlation_matrix = top_sector_pivot.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(top_correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Top Correlated Sectors')
plt.show()

# Plot time series and regression lines for the most correlated pair
top_pair = top_correlated_pairs.iloc[0]
sector1_data = data[data['Sector'] == top_pair['Sector1']]
sector2_data = data[data['Sector'] == top_pair['Sector2']]

plt.figure(figsize=(14, 8))

# Sector 1
plt.subplot(1, 2, 1)
plt.plot(sector1_data['Year'], sector1_data['FDI Inflows'], marker='o', label=f'{top_pair["Sector1"]} FDI Inflows')
plt.plot(sector1_data['Year'], np.poly1d(np.polyfit(sector1_data['Year'], sector1_data['FDI Inflows'], 1))(sector1_data['Year']), color='red', label='Regression Line')
plt.title(f'{top_pair["Sector1"]}')
plt.xlabel('Year')
plt.ylabel('FDI Inflows')
plt.legend()

# Sector 2
plt.subplot(1, 2, 2)
plt.plot(sector2_data['Year'], sector2_data['FDI Inflows'], marker='o', label=f'{top_pair["Sector2"]} FDI Inflows')
plt.plot(sector2_data['Year'], np.poly1d(np.polyfit(sector2_data['Year'], sector2_data['FDI Inflows'], 1))(sector2_data['Year']), color='red', label='Regression Line')
plt.title(f'{top_pair["Sector2"]}')
plt.xlabel('Year')
plt.ylabel('FDI Inflows')
plt.legend()

plt.tight_layout()
plt.show()

#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Pivot the data to have sectors as columns and years as index
sector_pivot = data.pivot(index='Year', columns='Sector', values='FDI Inflows')
correlation_matrix = sector_pivot.corr()

# Unstack the correlation matrix and convert it to a DataFrame
correlation_pairs = correlation_matrix.unstack()
correlation_pairs = pd.DataFrame(correlation_pairs)
index_list = correlation_pairs.index.to_list()

# Create lists from the index and values
sector1 = [index_list[i][0] for i in range(len(index_list))]
sector2 = [index_list[i][1] for i in range(len(index_list))]
corr_values = correlation_pairs[0].to_list()

# Create a DataFrame
correlation_pairs = pd.DataFrame({
    'Sector1': sector1,
    'Sector2': sector2,
    'Correlation': corr_values
})

# Filter out self-correlation and drop duplicate pairs
correlation_pairs = correlation_pairs[correlation_pairs['Sector1'] != correlation_pairs['Sector2']]
correlation_pairs = correlation_pairs.drop_duplicates(subset=['Sector1', 'Sector2'])

# Identify the top 5 pairs of sectors with the highest correlation
top_correlated_pairs = correlation_pairs.sort_values(by='Correlation', ascending=False).head(5)

print(top_correlated_pairs)

# Extract the unique sectors from the top correlated pairs
top_sectors = pd.unique(top_correlated_pairs[['Sector1', 'Sector2']].values.ravel('K'))

# Display the correlation matrix for the top sectors
top_sector_pivot = sector_pivot[top_sectors]
top_correlation_matrix = top_sector_pivot.corr()

plt.figure(figsize=(20, 15))
sns.heatmap(top_correlation_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 6})
plt.title('Correlation Matrix of Top Correlated Sectors')
plt.show()

# Plot time series and regression lines for the top 5 correlated pairs
for idx, row in top_correlated_pairs.iterrows():
    sector1_data = data[data['Sector'] == row['Sector1']]
    sector2_data = data[data['Sector'] == row['Sector2']]

    plt.figure(figsize=(14, 8))

    # Plot both sectors on the same plot
    plt.plot(sector1_data['Year'], sector1_data['FDI Inflows'], marker='o', label=f'{row["Sector1"]} FDI Inflows')
    plt.plot(sector2_data['Year'], sector2_data['FDI Inflows'], marker='o', label=f'{row["Sector2"]} FDI Inflows')

    # Plot regression lines for both sectors
    plt.plot(sector1_data['Year'], np.poly1d(np.polyfit(sector1_data['Year'], sector1_data['FDI Inflows'], 1))(sector1_data['Year']), color='red', label=f'{row["Sector1"]} Regression Line')
    plt.plot(sector2_data['Year'], np.poly1d(np.polyfit(sector2_data['Year'], sector2_data['FDI Inflows'], 1))(sector2_data['Year']), color='blue', label=f'{row["Sector2"]} Regression Line')

    plt.title(f'Time Series and Regression for {row["Sector1"]} and {row["Sector2"]}')
    plt.xlabel('Year')
    plt.ylabel('FDI Inflows')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# Prepare data for clustering
sector_data = sector_pivot.fillna(0)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sector_data.T)

# Perform k-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Add cluster labels to the data
sector_clusters = pd.DataFrame({'Sector': sector_pivot.columns, 'Cluster': labels})

# Print sectors in each cluster
for cluster_num in range(kmeans.n_clusters):
    print(f"Cluster {cluster_num + 1}:")
    print(sector_clusters[sector_clusters['Cluster'] == cluster_num]['Sector'].tolist())
    print()

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

plt.figure(figsize=(14, 8))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=labels, palette='viridis', s=100)

# Mark clusters with circles and label only once
for cluster_num in range(kmeans.n_clusters):
    cluster_points = pca_components[labels == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_num + 1}')
    cluster_center = cluster_points.mean(axis=0)
    plt.text(cluster_center[0], cluster_center[1], f'Cluster {cluster_num + 1}', fontsize=12, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

# Add a legend
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, title='Cluster')

plt.title('PCA of Sectors with K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()

print(sector_clusters)
