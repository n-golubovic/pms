import tkinter as tk
from tkinter import ttk
from pymilvus import Collection, connections, utility
import logging
import requests
from PIL import Image, ImageTk
from io import BytesIO
from functools import lru_cache
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

API_KEY = 'ab58eaf4'
OMDB_API_URL = f'https://www.omdbapi.com/'


def get_image_url(imdb_id):
    movie_data = get_movie_data(imdb_id)
    return movie_data.get('Poster', '')


@lru_cache(maxsize=100)
def get_movie_data(imdb_id):
    logger.info(f"Fetching movie data for {imdb_id}")
    params = {
        'i': imdb_id,
        'apikey': API_KEY
    }
    response = requests.get(OMDB_API_URL, params=params)
    response.raise_for_status()
    return response.json()


def fetch_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


class MovieSearchApp:
    def __init__(self, root):
        self.root = root

        self.search_results = []
        self.vector_columns = ['Comedy', 'Drama', 'Documentary', 'Romance', 'Horror', 'Action', 'Thriller', 'Family',
                               'Adventure', 'Crime', 'Science Fiction']
        self.genres = ['Comedy', 'Drama', 'Documentary', 'Romance', 'Horror', 'Action', 'Thriller', 'Family',
                       'Adventure', 'Crime', 'Science Fiction']
        self.image_label = None
        self.similar_movies = []
        self.chart_canvas = None

        # initialize UI elements
        self.chart_frame = None
        self.chart_label = None
        self.search_frame = None
        self.search_entry = None
        self.search_button = None
        self.results_frame = None
        self.results_list = None
        self.scrollbar = None
        self.details_frame = None
        self.image_frame = None
        self.info_frame = None
        self.movie_image = None
        self.movie_title = None
        self.movie_genres = None
        self.movie_overview = None
        self.recommendations_frame = None
        self.recommended_movies = None

        root.title("Movie Search App")

        try:
            connections.connect("default", host='localhost', port='19530')
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")

        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        collection_name = "movie_collection"
        if collection_name in utility.list_collections():
            self.collection = Collection(name=collection_name)
        else:
            logger.error(f"Collection {collection_name} not found.")

        # Left Frame for Search and Search Results
        self.left_frame = ttk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Middle Frame for Movie Details and Recommendations
        self.middle_frame = ttk.Frame(root)
        self.middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Frame for Genre Chart
        self.right_frame = ttk.Frame(root)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.setup_left_frame()
        self.setup_middle_frame()
        self.setup_right_frame()

    def setup_right_frame(self):
        self.chart_frame = ttk.Frame(self.right_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chart_label = ttk.Label(self.chart_frame, text="Genre Chart")
        self.chart_label.pack()

    def setup_left_frame(self):
        # Search Frame
        self.search_frame = ttk.Frame(self.left_frame)
        self.search_frame.pack(fill=tk.X, padx=10, pady=10)

        self.search_entry = ttk.Entry(self.search_frame)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.search_button = ttk.Button(self.search_frame, text="Search", command=self.perform_search)
        self.search_button.pack(side=tk.RIGHT)

        # Results Frame
        self.results_frame = ttk.Frame(self.left_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.results_list = tk.Listbox(self.results_frame, height=10)
        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_list.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_list.config(yscrollcommand=self.scrollbar.set)
        self.results_list.bind('<<ListboxSelect>>', self.show_movie_details)

    def setup_middle_frame(self):
        # Details Frame
        self.details_frame = ttk.Frame(self.middle_frame)
        self.details_frame.pack(fill=tk.BOTH, expand=True)

        # Image Frame
        self.image_frame = ttk.Frame(self.details_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Info Frame
        self.info_frame = ttk.Frame(self.details_frame)
        self.info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Movie Image Label
        self.movie_image = ttk.Label(self.image_frame, text="Movie Image")
        self.movie_image.pack(fill=tk.BOTH, expand=True)

        # Movie Title
        self.movie_title = ttk.Label(self.info_frame, text="Title: N/A")
        self.movie_title.pack(anchor='nw')

        # Movie Genres
        self.movie_genres = ttk.Label(self.info_frame, text="Genres: N/A")
        self.movie_genres.pack(anchor='nw')

        # Overview Frame
        self.overview_frame = ttk.Frame(self.info_frame)
        self.overview_frame.pack(fill=tk.BOTH, expand=True)

        # Movie Overview
        self.movie_overview = tk.Text(self.overview_frame, height=6, wrap='word')
        self.movie_overview.pack(fill=tk.BOTH, expand=True)
        self.movie_overview.config(state=tk.DISABLED)  # Make it read-only

        # Recommendations Frame
        self.recommendations_frame = ttk.Frame(self.middle_frame)
        self.recommendations_frame.pack(fill=tk.BOTH, expand=True)

        # Recommended Movies
        self.recommended_movies = ttk.Label(self.recommendations_frame, text="Recommended Movies", anchor="w")
        self.recommended_movies.pack(fill=tk.X)

    def perform_search(self):
        search_term = self.search_entry.get()
        results = self.search_movie_by_title(search_term)
        self.search_results = results
        self.results_list.delete(0, tk.END)

        for result in results:
            self.results_list.insert(tk.END, f"{result['title']}")

    def show_movie_details(self, event):
        selection = event.widget.curselection()

        if selection:
            index = selection[0]
            movie_data = self.search_results[index]
            self.update_movie_details(movie_data)

    def update_movie_details(self, movie_data):
        title = movie_data['title']
        features = movie_data['features']
        filtered_dict = {k: v for k, v in features.items() if k in self.genres}
        final_dict = {k: v for k, v in filtered_dict.items() if v > 0.5}
        self.load_image(movie_data['imdb_id'])
        genres = list(final_dict.keys())

        self.movie_overview.config(state=tk.NORMAL)
        self.movie_overview.delete('1.0', tk.END)
        self.movie_overview.insert(tk.END, movie_data['overview'])
        self.movie_overview.config(state=tk.DISABLED)

        self.draw_genre_chart(filtered_dict)

        self.movie_title.config(text=f"Title: {title}")
        self.movie_genres.config(text=f"Genres: {', '.join(genres)}")
        results = self.get_similar_movies(movie_data['features_vec'])

        self.update_recommendations(results)

    def on_closing(self):
        connections.disconnect("default")
        self.root.destroy()

    def update_recommendations(self, recommendations):
        for widget in self.recommendations_frame.winfo_children():
            if isinstance(widget, ttk.Label) and widget != self.recommended_movies:
                widget.destroy()

        for res in recommendations:
            formated_distance = "{:.2f}".format(res.distance)
            label_text = f"Distance: {formated_distance} - {res.entity.title}"

            label = ttk.Label(self.recommendations_frame, text=label_text, cursor="hand2")
            label.pack(fill=tk.X, expand=True)

            label.bind('<Button-1>', lambda e, rec=res: self.update_movie_details_from_rec(rec))

    def update_movie_details_from_rec(self, recommendation):
        if recommendation and recommendation.entity:
            entity = recommendation.entity

            features = {key: value for key, value in zip(self.vector_columns, entity.features)}

            self.update_movie_details({
                'title': entity.title,
                'features': features,
                'imdb_id': entity.imdb_id,
                'features_vec': entity.features,
                'overview': entity.overview
            })

    def search_movie_by_title(self, search_title):
        search_title_lower = search_title.lower()

        query_expression = f"title_lowercase like '%{search_title_lower}%'"
        output_fields = ["title", 'features', 'imdb_id', 'overview', 'tagline', 'transformed_genres']
        result = self.collection.query(
            expr=query_expression,
            output_fields=output_fields,
            limit=30
        )

        for res in result:
            print(f"Movie Found: {res['title']}")
            res['features_vec'] = res['features']
            result_dict = {key: value for key, value in zip(self.vector_columns, res['features'])}
            res['features'] = result_dict

        return result

    def get_similar_movies(self, query_vector):
        output_fields = ["title", 'features', 'imdb_id', 'overview', 'tagline', 'transformed_genres']
        search_params = {"metric_type": "L2", "offset": 1, "limit": 4}
        results = self.collection.search(data=[query_vector], anns_field="features", param=search_params,
                                         output_fields=output_fields, limit=4)
        self.similar_movies = results[0]
        return results[0]

    def load_image(self, movie_id):
        image_url = get_image_url(movie_id)
        image = fetch_image(image_url)
        photo = ImageTk.PhotoImage(image)
        self.movie_image.config(image=photo)
        self.movie_image.image = photo

    def draw_genre_chart(self, genres):
        if self.chart_canvas is not None:
            self.chart_canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(6, 4), dpi=100)
        plot = fig.add_subplot(111)

        genre_names = list(genres.keys())
        genre_values = list(genres.values())
        y_ticks = range(len(genre_names))

        plot.barh(y_ticks, genre_values, color='blue')

        fig.subplots_adjust(left=0.2)

        plot.set_xlabel('Score')
        plot.set_yticks(y_ticks)
        plot.set_yticklabels(genre_names, rotation=0, fontsize=10)

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas_widget = self.chart_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.chart_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieSearchApp(root)
    root.mainloop()
