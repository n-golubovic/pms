from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import connections, utility

import pandas as pd

connections.connect("default", host="localhost", port="19530")


def create_collection():
    vector_field = FieldSchema(name="features", dtype=DataType.FLOAT_VECTOR, dim=11,
                               description="Combined feature vector including genres and vote metrics")

    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="Primary ID")
    imbd_id_field = FieldSchema(name="imdb_id", dtype=DataType.VARCHAR, description="IMDB ID string", max_length=25)
    overview_field = FieldSchema(name="overview", dtype=DataType.VARCHAR, description="Movie overview", max_length=2000)
    title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, description="Movie title", max_length=500)
    title_lowercase_field = FieldSchema(name="title_lowercase", dtype=DataType.VARCHAR, description="Lowercase title",max_length=500)
    tagline_field = FieldSchema(name="tagline", dtype=DataType.VARCHAR, description="Movie tagline", max_length=500)
    transformed_genres_field = FieldSchema(name="transformed_genres", dtype=DataType.VARCHAR,description="Transformed genres", max_length=500)

    schema = CollectionSchema(
        fields=[
            id_field,
            imbd_id_field,
            overview_field,
            title_field,
            title_lowercase_field,
            tagline_field,
            transformed_genres_field,
            vector_field
        ], description="Movie data schema")

    collection_name = "movie_collection"
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }

    # Create the index
    collection.create_index(field_name="features", index_params=index_params)

    print(f"Collection {collection_name} created successfully.")


def drop_collection(collection_name):
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)

        collection.drop()

        print(f"Collection '{collection_name}' has been dropped.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


def insert_data(df, collection_name):
    vector_columns = ['vote_average', 'vote_count', 'Comedy', 'Drama', 'Documentary', 'Romance', 'Horror', 'Action', 'Thriller', 'Family', 'Adventure', 'Crime', 'Science Fiction']
    df[vector_columns] = df[vector_columns].fillna(0)
    scalar_columns = ['imdb_id', 'overview', 'title', 'title_lowercase', 'tagline', 'transformed_genres']
    df[scalar_columns] = df[scalar_columns].fillna('')

    df['vector'] = df[['Comedy', 'Drama', 'Documentary', 'Romance', 'Horror', 'Action', 'Thriller', 'Family', 'Adventure', 'Crime', 'Science Fiction']].values.tolist()
    data_to_insert = df.apply(lambda row: {
        'features': row['vector'],
        'overview': row['overview'],
        'title': row['title'],
        'title_lowercase': row['title_lowercase'],
        'tagline': row['tagline'],
        'transformed_genres': row['transformed_genres'],
        'imdb_id': row['imdb_id']
    }, axis=1).tolist()

    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
    else:
        print(f"Collection {collection_name} does not exist.")
        return

    # Insert data into Milvus
    mr = collection.insert(data=data_to_insert)

    collection.load()
    print(f"Data inserted into {collection_name} with auto-generated IDs.")


def view_schema(collection_name):
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        schema = collection.schema
        print("Schema of the collection:")
        print(schema)
    else:
        print(f"Collection '{collection_name}' does not exist.")


drop_collection("movie_collection")
create_collection()

view_schema("movie_collection")

df = pd.read_csv('movie_data_cleaned.csv')
insert_data(df, "movie_collection")

collection_name = "movie_collection"

connections.disconnect("default")
