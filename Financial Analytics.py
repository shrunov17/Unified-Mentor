# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 23:35:52 2024

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
# Load the dataset
file_path = r'D:\IIT DELHI Material\Prrrrooojjjecccttt\unified mentor\Financial Analytics data.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Data Cleaning (if necessary)
# Example: data.dropna(inplace=True)

# Top Companies by Market Capitalization
top_market_cap = data.sort_values(by='Mar Cap - Crore', ascending=False).head(10)
print("Top 10 Companies by Market Capitalization:")
print(top_market_cap)

# Top Companies by Quarterly Sales
top_sales = data.sort_values(by='Sales Qtr - Crore', ascending=False).head(10)
print("Top 10 Companies by Quarterly Sales:")
print(top_sales)

# Distribution Analysis
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['Mar Cap - Crore'], kde=True, bins=30)
plt.title('Market Capitalization Distribution')

plt.subplot(1, 2, 2)
sns.histplot(data['Sales Qtr - Crore'], kde=True, bins=30)
plt.title('Quarterly Sales Distribution')
plt.show()

# Correlation Analysis between Market Capitalization and Quarterly Sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sales Qtr - Crore', y='Mar Cap - Crore', data=data)
plt.title('Market Capitalization vs Quarterly Sales')
plt.xlabel('Quarterly Sale in Crores')
plt.ylabel('Market Capitalization in Crores')
plt.show()

# Outliers Detection in Market Capitalization and Quarterly Sales
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y='Mar Cap - Crore', data=data)
plt.title('Market Capitalization Outliers')

plt.subplot(1, 2, 2)
sns.boxplot(y='Sales Qtr - Crore', data=data)
plt.title('Quarterly Sales Outliers')
plt.show()

# Ratio Analysis: Market Cap to Sales Ratio
data['Market Cap to Sales Ratio'] = data['Mar Cap - Crore'] / data['Sales Qtr - Crore']
top_ratios = data.sort_values(by='Market Cap to Sales Ratio', ascending=False).head(10)
print("Top 10 Companies by Market Cap to Sales Ratio:")
print(top_ratios)

# Normalize each parameter individually
scaler = MinMaxScaler()
top_ratios['Mar Cap - Crore'] = scaler.fit_transform(top_ratios[['Mar Cap - Crore']])
top_ratios['Sales Qtr - Crore'] = scaler.fit_transform(top_ratios[['Sales Qtr - Crore']])
top_ratios['Market Cap to Sales Ratio'] = scaler.fit_transform(top_ratios[['Market Cap to Sales Ratio']])

# Generate a bar graph for the top 10 companies by Market Cap to Sales Ratio
plt.figure(figsize=(14, 8))
bar_width = 0.25
bar_l = list(range(len(top_ratios['Name'])))
tick_pos = [i + (bar_width / 2) for i in bar_l]

plt.bar(bar_l, top_ratios['Mar Cap - Crore'], width=bar_width, label='Market Cap (Normalized)', color='b')
plt.bar([p + bar_width for p in bar_l], top_ratios['Sales Qtr - Crore'], width=bar_width, label='Sales (Normalized)', color='g')
plt.bar([p + bar_width*2 for p in bar_l], top_ratios['Market Cap to Sales Ratio'], width=bar_width, label='Market Cap to Sales Ratio (Normalized)', color='r')

plt.xlabel('Company')
plt.ylabel('Normalized Value')
plt.title('Normalized Market Cap, Sales, and Market Cap to Sales Ratio of Top 10 Companies by Ratio')
plt.xticks(tick_pos, top_ratios['Name'], rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Manually create a dictionary of company names and their ticker symbols
manual_tickers = {
    'Reliance Industries Limited': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'Infosys Limited': 'INFY.NS',
    'HDFC Bank Limited': 'HDFCBANK.NS',
    'ICICI Bank Limited': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Bharti Airtel Limited': 'BHARTIARTL.NS',
    'Hindustan Unilever Limited': 'HINDUNILVR.NS',
    'Kotak Mahindra Bank Limited': 'KOTAKBANK.NS',
    'ITC Limited': 'ITC.NS',
    'Larsen & Toubro Limited': 'LT.NS',
    'Axis Bank Limited': 'AXISBANK.NS',
    'Bajaj Finance Limited': 'BAJFINANCE.NS',
    'Maruti Suzuki India Limited': 'MARUTI.NS',
    'Sun Pharmaceutical Industries Limited': 'SUNPHARMA.NS',
    'Wipro Limited': 'WIPRO.NS',
    'HCL Technologies Limited': 'HCLTECH.NS',
    'Asian Paints Limited': 'ASIANPAINT.NS',
    'Nestle India Limited': 'NESTLEIND.NS',
    'Tata Steel Limited': 'TATASTEEL.NS',
    'Mahindra & Mahindra Limited': 'M&M.NS',
    'Hindalco Industries Limited': 'HINDALCO.NS',
    'Tata Motors Limited': 'TATAMOTORS.NS',
    'UltraTech Cement Limited': 'ULTRACEMCO.NS',
    'SBI Life Insurance Company Limited': 'SBILIFE.NS',
    'Tech Mahindra Limited': 'TECHM.NS',
    'Britannia Industries Limited': 'BRITANNIA.NS',
    'Dr. Reddy\'s Laboratories Limited': 'DRREDDY.NS',
    'Divi\'s Laboratories Limited': 'DIVISLAB.NS',
    'Adani Ports and Special Economic Zone Limited': 'ADANIPORTS.NS',
    'Power Grid Corporation of India Limited': 'POWERGRID.NS',
    'Eicher Motors Limited': 'EICHERMOT.NS',
    'HDFC Life Insurance Company Limited': 'HDFCLIFE.NS',
    'Hindustan Zinc Limited': 'HINDZINC.NS',
    'JSW Steel Limited': 'JSWSTEEL.NS',
    'Grasim Industries Limited': 'GRASIM.NS',
    'Indian Oil Corporation Limited': 'IOC.NS',
    'Bharat Petroleum Corporation Limited': 'BPCL.NS',
    'Oil and Natural Gas Corporation Limited': 'ONGC.NS',
    'Coal India Limited': 'COALINDIA.NS',
    'Hero MotoCorp Limited': 'HEROMOTOCO.NS',
    'Bajaj Auto Limited': 'BAJAJ-AUTO.NS',
    'GAIL (India) Limited': 'GAIL.NS',
    'Cipla Limited': 'CIPLA.NS',
    'Shree Cement Limited': 'SHREECEM.NS',
    'Bharti Infratel Limited': 'INFRATEL.NS',
    'Godrej Consumer Products Limited': 'GODREJCP.NS',
    'Pidilite Industries Limited': 'PIDILITIND.NS',
    'Torrent Pharmaceuticals Limited': 'TORNTPHARM.NS'
}

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['Company Name', 'Ticker Symbol', 'Market Capitalization', 'Quarterly Sales', 'Market Cap to Sales Ratio', 'Start Date Closing Price', 'End Date Closing Price', 'Percentage Change'])

# Specify the period for financial data and stock prices
start_date = '2023-01-01'
end_date = '2023-12-31'

# Fetch market capitalization, quarterly sales, and stock prices for each ticker
for company_name, ticker in manual_tickers.items():
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch market capitalization
        market_cap = stock.info['marketCap']
        
        # Fetch quarterly financials
        quarterly_financials = stock.quarterly_financials
        recent_quarter = quarterly_financials.columns[0]
        quarterly_sales = quarterly_financials.loc['Total Revenue'][recent_quarter]
        
        # Calculate market cap to sales ratio
        market_cap_to_sales_ratio = market_cap / quarterly_sales
        
        # Fetch closing prices
        historical_prices = stock.history(start=start_date, end=end_date)
        start_price = historical_prices['Close'].iloc[0]
        end_price = historical_prices['Close'].iloc[-1]
        
        # Calculate percentage change in stock price
        percentage_change = ((end_price - start_price) / start_price) * 100
        
        # Append the results to the DataFrame
        new_row = pd.DataFrame({
            'Company Name': [company_name],
            'Ticker Symbol': [ticker],
            'Market Capitalization': [market_cap],
            'Quarterly Sales': [quarterly_sales],
            'Market Cap to Sales Ratio': [market_cap_to_sales_ratio],
            'Start Date Closing Price': [start_price],
            'End Date Closing Price': [end_price],
            'Percentage Change': [percentage_change]
        })
        
        results = pd.concat([results, new_row], ignore_index=True)
        
    except Exception as e:
        print(f"Error fetching data for {company_name} ({ticker}): {e}")

# Ensure the columns are numeric
results['Market Capitalization'] = pd.to_numeric(results['Market Capitalization'], errors='coerce')
results['Quarterly Sales'] = pd.to_numeric(results['Quarterly Sales'], errors='coerce')
results['Market Cap to Sales Ratio'] = pd.to_numeric(results['Market Cap to Sales Ratio'], errors='coerce')
results['Percentage Change'] = pd.to_numeric(results['Percentage Change'], errors='coerce')

# Drop rows with NaN values
results.dropna(inplace=True)

# Apply log transformation to spread out the values
results['Log Market Cap to Sales Ratio'] = np.log(results['Market Cap to Sales Ratio'] + 1)  # Adding 1 to avoid log(0)

# Display the results
print(results)

# Save the results to a CSV file
# results.to_csv('financial_data_with_tickers_and_ratios.csv', index=False)

# Calculate correlations
correlation_market_cap = results[['Market Capitalization', 'Percentage Change']].corr().iloc[0, 1]
correlation_sales = results[['Quarterly Sales', 'Percentage Change']].corr().iloc[0, 1]
correlation_ratio = results[['Market Cap to Sales Ratio', 'Percentage Change']].corr().iloc[0, 1]

print(f"Correlation between Market Cap and Percentage Change: {correlation_market_cap:.2f}")
print(f"Correlation between Quarterly Sales and Percentage Change: {correlation_sales:.2f}")
print(f"Correlation between Market Cap to Sales Ratio and Percentage Change: {correlation_ratio:.2f}")

# Plot the correlations
plt.figure(figsize=(18, 6))

# Market Cap vs Percentage Change
# plt.subplot(1, 3, 1)
sns.regplot(x='Market Capitalization', y='Percentage Change', data=results, scatter_kws={'s':20}, line_kws={'color':'red'})
plt.title('Market Cap vs Percentage Change')
plt.xlabel('Market Capitalization')
plt.ylabel('Percentage Change')
plt.tight_layout()
plt.show()
# Quarterly Sales vs Percentage Change
plt.figure(figsize=(18, 6))
sns.regplot(x='Quarterly Sales', y='Percentage Change', data=results, scatter_kws={'s':20}, line_kws={'color':'red'})
plt.title('Quarterly Sales vs Percentage Change')
plt.xlabel('Quarterly Sales')
plt.ylabel('Percentage Change')
plt.tight_layout()
plt.show()
# Market Cap to Sales Ratio vs Percentage Change
plt.figure(figsize=(18, 6))
sns.regplot(x='Log Market Cap to Sales Ratio', y='Percentage Change', data=results, scatter_kws={'s':20}, line_kws={'color':'red'})
plt.title('Market Cap to Sales Ratio vs Percentage Change')
plt.xlabel('Log Market Cap to Sales Ratio')
plt.ylabel('Percentage Change')

plt.tight_layout()
plt.show()
