import numpy as np
import tensorflow as tf
from preprocess import load_data, preprocess_data
from embeddings import load_glove_embeddings
from model import create_model
from download_glove import download_glove_embeddings

# Download GloVe embeddings
embedding_dim = 100
download_glove_embeddings(embedding_dim)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = load_data()
x_train_pad, x_test_pad, tokenizer = preprocess_data(x_train, x_test)

# Load GloVe embeddings
embedding_matrix = load_glove_embeddings(tokenizer.word_index, embedding_dim)

# Create and train the model
vocab_size = len(tokenizer.word_index) + 1
max_len = x_train_pad.shape[1]

model = create_model(vocab_size, embedding_dim, embedding_matrix, max_len)

history = model.fit(x_train_pad, y_train, 
                    validation_split=0.2, 
                    epochs=10, 
                    batch_size=64)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_pad, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Function to predict sentiment
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

# Example predictions
print(predict_sentiment("This movie was fantastic! I really enjoyed it."))
print(predict_sentiment("I didn't like this film at all. It was boring and predictable."))

