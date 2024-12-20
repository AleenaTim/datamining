import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'TMDB_movie_dataset_v11.csv'
df = pd.read_csv(file_path)

columns_to_drop = ['id', 'status', 'adult', 'imdb_id', 'original_language', 
                   'original_title', 'overview', 'poster_path', 'production_companies',
                   'production_countries', 'spoken_languages', 'backdrop_path', 'homepage', 'tagline', 'keywords']
print(f"Dropping {len(columns_to_drop)} unnecessary columns: {columns_to_drop}")
df = df.drop(columns=columns_to_drop)

columns_to_check = ['vote_count', 'popularity', 'revenue', 'budget']
before_rows = len(df)
df = df[(df[columns_to_check] != 0.0).all(axis=1)]
print(f"Removed {before_rows - len(df)} rows with 0.0 values in columns: {columns_to_check}")

if df.duplicated().sum() > 0:
    print(f"\nFound {df.duplicated().sum()} duplicate rows. Removing them.")
    df = df.drop_duplicates()

df['genres'] = df['genres'].str.split(', ')
unique_genres = sorted(set(genre for sublist in df['genres'].dropna() for genre in sublist))  # Get all unique genres
print(f"Performing one-hot encoding for {len(unique_genres)} unique genres: {unique_genres}")

for genre in unique_genres:
    df[f'{genre}'] = df['genres'].apply(lambda x: 1 if isinstance(x, list) and genre in x else 0)

print("Dropping the original 'genres' column.")
df = df.drop(columns=['genres'])

# Sort by popularity
print("Sorting the dataset by popularity in descending order.")
df = df.sort_values(by='popularity', ascending=False)

# Split into training and test sets
print("Splitting the dataset into training and test sets (80-20 split).")
df_working, df_final_test = train_test_split(df, test_size=0.2, random_state=42)

# Save the datasets
test_set_path = 'test_set.csv'
working_set_path = 'training_set.csv'

df_final_test.to_csv(test_set_path, index=False)
df_working.to_csv(working_set_path, index=False)

print(f"\nFinal test set saved to {test_set_path}. Shape: {df_final_test.shape}")
print(f"Cleaned working dataset saved to {working_set_path}. Shape: {df_working.shape}")
