import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gzip
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
import time

file_path = 'movies_metadata_filtered.csv'

data = pd.read_csv(file_path, low_memory=False)

data['genres'] = data['transformed_genres'].apply(lambda x: ast.literal_eval(x))

data.head()

train, temp = train_test_split(data, test_size=0.2, random_state=23)
val, test = train_test_split(temp, test_size=0.5, random_state=23)


data = pd.concat([train, val])

GENRES_2 = ['Comedy', 'Drama', 'Documentary', 'Romance', 'Horror', 'Action', 'Thriller', 'Family', 'Adventure',
            'Crime', 'Science Fiction']
data['genres'] = data['genres'].apply(lambda x: [genre for genre in x if genre in GENRES_2])
data = data[data['genres'].apply(len) > 0]
train, val = train_test_split(data, test_size=0.2, random_state=42)

train['overview_compressed'] = train['overview'].apply(lambda x: len(gzip.compress(x.encode())))

def get_multilabel_data(overview):
    x1 = overview
    print(x1)
    cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []

    training_set_idx = []
    genres_idx = []

    for train_index, train_row in tqdm(train.iterrows(), total=len(train), desc="Processing training set", unit="row"):
        x2 = train_row.overview
        cx2 = train_row.overview_compressed
        x1x2 = " ".join([x1, x2])
        cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (cx1x2 - min(cx1 ,cx2)) / max(cx1 , cx2)
        distance_from_x1.append(ncd)
        training_set_idx.append(train_index)
        genres_idx.append(train_row.genres)

    return distance_from_x1, training_set_idx, genres_idx


val_predictions = []
val_predictions_binary = []

# print number of rows in val
print(len(val))

# log start time
start_time = time.time()
counter = 0

for idx, row in val.iterrows():
    distances, rows, genres = get_multilabel_data(row.overview)

    mlb = MultiLabelBinarizer(classes=GENRES_2)
    y_train_binarized = mlb.fit_transform(genres)
    x_train = np.array(distances).reshape(-1, 1)

    mlknn = MLkNN(k=10)
    mlknn.fit(x_train, y_train_binarized)

    X_test = np.array([[0]])
    y_pred = mlknn.predict(X_test)

    predicted_genres = mlb.inverse_transform(y_pred)

    y_pred = y_pred.toarray()[0]

    val_predictions.append(predicted_genres)
    val_predictions_binary.append(y_pred)

    counter += 1

    if counter == 1:
        process_time = time.time() - start_time
        print(f"Time taken for 1 row: {process_time}")

    break

mlb = MultiLabelBinarizer(classes=GENRES_2)
val_genres_binarized = mlb.fit_transform(val['genres'])

val_predictions_binary1 = np.array(val_predictions_binary)

# take frist element of val predictions binary of length val_predictions_binary1
val_genres_binarized1 = val_genres_binarized[:len(val_predictions_binary1)]

# print first val genres binarized
print(val_predictions_binary1)
print(val_genres_binarized1)
