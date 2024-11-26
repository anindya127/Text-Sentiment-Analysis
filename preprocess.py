import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, x_test, max_len=200):
    # Convert integer sequences to text
    word_index = tf.keras.datasets.imdb.get_word_index()
    index_word = {i+3: word for word, i in word_index.items()}
    index_word[0] = '<PAD>'
    index_word[1] = '<START>'
    index_word[2] = '<UNK>'

    x_train_text = [' '.join([index_word.get(i, '<UNK>') for i in sequence]) for sequence in x_train]
    x_test_text = [' '.join([index_word.get(i, '<UNK>') for i in sequence]) for sequence in x_test]

    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train_text)

    x_train_seq = tokenizer.texts_to_sequences(x_train_text)
    x_test_seq = tokenizer.texts_to_sequences(x_test_text)

    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding='post', truncating='post')
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding='post', truncating='post')

    return x_train_pad, x_test_pad, tokenizer

