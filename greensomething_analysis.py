


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
# Load the dataset
file_path = r'D:\IIT DELHI Material\Prrrrooojjjecccttt\unified mentor\greendestination (1).csv'  # Update this to your file path  # Update this to your file path
data = pd.read_csv(file_path)

# Calculate the attrition rate
attrition_rate = data['Attrition'].value_counts(normalize=True)['Yes'] * 100
print(f"Attrition Rate: {attrition_rate:.2f}%")

# Function to plot Gaussian distribution with statistical parameters
def plot_gaussian(data, column, attrition_status):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, stat="density", linewidth=0, label="Data")
    mu, std = norm.fit(data[column])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label="Gaussian Fit")
    title = f'{column} Distribution for {attrition_status}\nMean: {mu:.2f}, Std Dev: {std:.2f}'
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters to analyze
parameters = ['Age', 'YearsAtCompany', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction']

# Plot distributions for each parameter based on attrition status
for param in parameters:
    plot_gaussian(data[data['Attrition'] == 'Yes'], param, 'Attrition')
    plot_gaussian(data[data['Attrition'] == 'No'], param, 'No Attrition')

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'D:\IIT DELHI Material\Prrrrooojjjecccttt\unified mentor\greendestination (1).csv'  # Update this to your file path  # Update this to your file path  # Update this to your file path
data = pd.read_csv(file_path)

# Convert 'Attrition' column to numeric
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Select all numerical columns
numerical_columns = data.select_dtypes(include='number').columns

# Calculate the correlation matrix for numerical columns
correlation_matrix = data[numerical_columns].corr()

# Extract correlations with 'Attrition'
attrition_correlation = correlation_matrix[['Attrition']]

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attrition_correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation with Attrition')
plt.show()

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'D:\IIT DELHI Material\Prrrrooojjjecccttt\unified mentor\greendestination (1).csv'  # Update this to your file path  # Update this to your file path  # Update this to your file path  # Update this to your file path
data = pd.read_csv(file_path)

# Convert 'Attrition' column to numeric
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Select all numerical columns
numerical_columns = data.select_dtypes(include='number').columns

# Calculate the correlation matrix for numerical columns
correlation_matrix = data[numerical_columns].corr()

# Extract correlations with 'Attrition'
attrition_correlation = correlation_matrix[['Attrition']].drop(index=['Attrition'])

# Plotting the heatmap with enhanced aesthetics
plt.figure(figsize=(12, 10))
sns.set(font_scale=1.2)
heatmap = sns.heatmap(attrition_correlation, annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='black')
heatmap.set_title('Correlation of Numerical Parameters with Attrition', fontsize=18)
plt.yticks(rotation=0)  # Rotate y-axis labels for better readability
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np

# Load the dataset
file_path = r'D:\IIT DELHI Material\Prrrrooojjjecccttt\unified mentor\greendestination (1).csv'  # Update this to your file path  # Update this to your file path  # Update this to your file path  # Update this to your file path  # Update this to your file path
data = pd.read_csv(file_path)

# Convert 'Attrition' column to numeric
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Calculate the attrition rate
attrition_rate = data['Attrition'].value_counts(normalize=True)[1] * 100
print(f"Attrition Rate: {attrition_rate:.2f}%")

# Select all numerical columns
numerical_columns = data.select_dtypes(include='number').columns

# Calculate the correlation matrix for numerical columns
correlation_matrix = data[numerical_columns].corr()

# Extract correlations with 'Attrition'
attrition_correlation = correlation_matrix['Attrition'].drop(labels=['Attrition'])

# Ensure the inclusion of specific parameters
parameters = set(['Age', 'YearsAtCompany', 'MonthlyIncome'])

# Select the top parameters based on absolute correlation values
top_parameters = attrition_correlation.abs().sort_values(ascending=False).index
for param in top_parameters:
    if len(parameters) >= 5:
        break
    parameters.add(param)

# Convert the set to a list
parameters = list(parameters)

# Function to plot Gaussian distribution with statistical parameters
def plot_gaussian(data, column, attrition_status):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, stat="density", linewidth=0, label="Data", color="skyblue", bins=20)
    mu, std = norm.fit(data[column])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label="Gaussian Fit", color="red")
    title = f'{column} Distribution for {attrition_status}\nMean: {mu:.2f}, Std Dev: {std:.2f}'
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot distributions for each parameter based on attrition status
for param in parameters:
    plot_gaussian(data[data['Attrition'] == 1], param, 'Attrition')
    plot_gaussian(data[data['Attrition'] == 0], param, 'No Attrition')

# Plotting the correlation matrix for the top parameters
relevant_correlation_matrix = correlation_matrix.loc[parameters, ['Attrition']]

plt.figure(figsize=(12, 8))
sns.set(font_scale=1.2)
heatmap = sns.heatmap(relevant_correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='black')
heatmap.set_title('Top Correlation of Numerical Parameters with Attrition', fontsize=18)
plt.yticks(rotation=0)  # Rotate y-axis labels for better readability
plt.show()

# Plotting the correlation matrix for all numerical parameters
# Plotting the correlation matrix for all numerical parameters with improved aesthetics
plt.figure(figsize=(15, 13))
sns.set(font_scale=0.8)
heatmap_all = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='black', fmt=".2f")
heatmap_all.set_title('Correlation Matrix of All Numerical Parameters', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()