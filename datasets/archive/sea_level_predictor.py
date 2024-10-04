import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the dataset
df = pd.read_csv(r'C:\Users\arman\OneDrive\Desktop\Coding\epa-sea-level.csv')

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], label='Data', color='blue')

# Line of best fit for the entire dataset
slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])

# Predict sea level for years up to 2050
years_extended = list(range(df['Year'].min(), 2051))
sea_level_extended = [slope * year + intercept for year in years_extended]

plt.plot(years_extended, sea_level_extended, color='red', linestyle='--', label='Best fit line (1880-2050)')

# Filter data from 2000 to the most recent year
df_recent = df[df['Year'] >= 2000]

# Line of best fit for the recent data
slope_recent, intercept_recent, _, _, _ = linregress(df_recent['Year'], df_recent['CSIRO Adjusted Sea Level'])

# Predict sea level for years up to 2050
sea_level_recent = [slope_recent * year + intercept_recent for year in years_extended]

plt.plot(years_extended, sea_level_recent, color='green', linestyle='-', label='Best fit line (2000-2050)')

# Add labels, title, and legend
plt.xlabel('Year')
plt.ylabel('Sea Level (inches)')
plt.title('Rise in Sea Level')
plt.legend()

# Save and show the plot
plt.savefig('sea_level_rise.png')
plt.show()
