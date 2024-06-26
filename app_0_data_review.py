import pandas as pd

df = pd.read_csv('movie_data.csv')

print(df.head())

original_df = df.copy()

df.fillna({
    'vote_average': 0,
    'vote_count': 0,
    'Comedy': 0,
    'Drama': 0,
    'Documentary': 0,
    'Romance': 0,
    'Horror': 0,
    'Action': 0,
    'Thriller': 0,
    'Family': 0,
    'Adventure': 0,
    'Crime': 0,
    'Science Fiction': 0,
    'imbd_id': '',
    'overview': '',
    'title': '',
    'title_lowercase': '',
    'tagline': '',
    'transformed_genres': ''
}, inplace=True)

changes = (original_df != df).any(axis=1).sum()
print(f'Number of rows with at least one change: {changes}')

df.to_csv('movie_data_cleaned.csv', index=False)
