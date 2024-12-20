import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training set
file_path = 'training_set.csv'
df = pd.read_csv(file_path)

# Popularity vs. Vote Average
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='vote_average', y='popularity', alpha=0.7)
plt.title('Popularity vs. Vote Average')
plt.xlabel('Vote Average')
plt.ylabel('Popularity')
plt.show()

# Binning Popularity into Broader Bins
bin_width = 50  # Increased bin width to reduce the number of bins
popularity_bins = range(0, int(df['popularity'].max()) + bin_width, bin_width)
df['popularity_bin'] = pd.cut(df['popularity'], bins=popularity_bins, right=False)

# Cap revenue outliers at 95th percentile to reduce skewness
revenue_cap = df['revenue'].quantile(0.95)
df['revenue'] = df['revenue'].clip(upper=revenue_cap)

# Apply log transformation to popularity to handle skewness
import numpy as np
df['popularity_log'] = np.log1p(df['popularity'])  # log(1 + x) to avoid issues with 0 values

# Plot the distribution of log-transformed popularity
plt.figure(figsize=(12, 6))
sns.histplot(df['popularity_log'], bins=30, kde=True, color='blue', alpha=0.6)
plt.title('Distribution of Log-Transformed Popularity', fontsize=18)
plt.xlabel('Log(Popularity)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Total Popularity per Genre
genre_columns = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 
                 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
                 'TV Movie', 'Thriller', 'War', 'Western']

genre_popularity = {}
for genre in genre_columns:
    genre_popularity[genre] = df[df[genre] == 1]['popularity'].sum()

genre_popularity_df = pd.DataFrame.from_dict(genre_popularity, orient='index', columns=['Total Popularity']).sort_values(by='Total Popularity', ascending=False)

plt.figure(figsize=(12, 6))
genre_popularity_df['Total Popularity'].plot(kind='bar', color='purple')
plt.title('Total Popularity by Genre')
plt.ylabel('Total Popularity')
plt.xlabel('Genre')
plt.xticks(rotation=45)
plt.show()

# Correlation between Popularity and Vote Average
popularity_vote_corr = df['popularity'].corr(df['vote_average'])
print(f"Correlation between Popularity and Vote Average: {popularity_vote_corr:.2f}")

# Correlation between Popularity and Revenue
popularity_revenue_corr = df['popularity'].corr(df['revenue'])
print(f"Correlation between Popularity and Revenue: {popularity_revenue_corr:.2f}")

print("\nFeature Exploration with Predictive and Statistical Insights Completed!")
