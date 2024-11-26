# Sentiment Analysis with GloVe Embeddings

This project performs sentiment analysis on movie reviews using GloVe embeddings and a neural network model built with TensorFlow.

## Project Structure

- `download_glove.py`: Script to download GloVe embeddings.
- `embeddings.py`: Functions to load GloVe embeddings.
- `main.py`: Main script to train the model and make predictions.
- `model.py`: Functions to create the neural network model.
- `preprocess.py`: Functions to load and preprocess data.
- `requirements.text`: List of dependencies.

## Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/anindya127/Text-Sentiment-Analysis.git
    ```

2. Install the dependencies:

    ```sh
    pip install -r requirements.text
    ```

## Usage

1. Train the model:

    ```sh
    python main.py
    ```

2. Predict sentiment:
    Modify the `predict_sentiment` function in `main.py` to input your text and run:

    ```sh
    python main.py
    ```

## Functions

### `download_glove.py`

- `download_glove_embeddings(embedding_dim=100)`: Downloads and extracts GloVe embeddings.

### `embeddings.py`

- `load_glove_embeddings(word_index, embedding_dim=100)`: Loads GloVe embeddings into an embedding matrix.

### `preprocess.py`

- `load_data()`: Loads the IMDB dataset.
- `preprocess_data(x_train, x_test, max_len=200)`: Preprocesses the data by tokenizing and padding sequences.

### `model.py`

- `create_model(vocab_size, embedding_dim, embedding_matrix, max_len)`: Creates and compiles the neural network model.

### `main.py`

- `predict_sentiment(text)`: Predicts the sentiment of the given text.

## License

This project is licensed under the MIT License.
